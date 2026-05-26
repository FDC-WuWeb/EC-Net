import os
import os.path as osp
from typing import Tuple, Dict
import yaml
import ipdb
import torch
import tqdm
import numpy as np
from geotransformer.engine.base_trainer import BaseTrainer
from geotransformer.utils.torch import to_cuda
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.timer import Timer
from geotransformer.utils.common import get_log_string
import argparse
import open3d as o3d
import time
from easydict import EasyDict as edict
from nets import Deformation_Pyramid
# from nets import
import torch.optim as optim

from typing import Union
import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
def _validate_chamfer_reduction_inputs(
        batch_reduction: Union[str, None], point_reduction: str
):
    """Check the requested reductions are valid.
    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(
        points: Union[torch.Tensor, Pointclouds],
        lengths: Union[torch.Tensor, None],
        normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
                lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


def compute_truncated_chamfer_distance(
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        weights=None,
        trunc=0.2,
        batch_reduction: Union[str, None] = "mean",
        point_reduction: str = "mean",
):

    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    # truncation
    x_mask[cham_x >= trunc] = True
    y_mask[cham_y >= trunc] = True
    cham_x[x_mask] = 0.0
    cham_y[y_mask] = 0.0

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    # Apply point reduction

    # cham_x = cham_x.sum(1)  # (N,)
    # cham_y = cham_y.sum(1)  # (N,)

    # use l1 norm, more robust to partial case
    cham_x = torch.sqrt(cham_x).sum(1)  # (N,)
    cham_y = torch.sqrt(cham_y).sum(1)  # (N,)

    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    # cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist


def remove_outliers(point_cloud):
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    return point_cloud.select_by_index(ind)


def remove_outliers_adaptive(point_cloud, neighbor, std_ratio=0.1):
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=neighbor, std_ratio=std_ratio)
    return point_cloud.select_by_index(ind)

def compute_chamfer_distance_adaptive(cloud1, cloud2):
    dist1 = cloud1.compute_point_cloud_distance(cloud2)
    dist2 = cloud2.compute_point_cloud_distance(cloud1)
    chamfer_dist = (np.mean(dist1) + np.mean(dist2)) / 2
    return chamfer_dist

def compute_similarity(cloud1, cloud2):
    return compute_chamfer_distance_adaptive(cloud1, cloud2)

def auto_tune_parameters(src_points, src_corr_points, ref_points, ref_corr_points, neighbor_range, threshold_range):
    best_threshold = None
    best_neighbor = None
    best_similarity = float('inf')
    best_cropped_src = None
    best_cropped_ref = None

    for threshold in threshold_range:
        distances = np.asarray(src_points.compute_point_cloud_distance(src_corr_points))
        cropped_indices = np.where(distances < threshold)[0]
        cropped_src = src_points.select_by_index(cropped_indices)

        for neighbor in neighbor_range:
            ref_corr_filtered = remove_outliers_adaptive(ref_corr_points, neighbor, std_ratio=0.1)

            distances_ref = np.asarray(ref_points.compute_point_cloud_distance(ref_corr_filtered))
            cropped_indices_ref = np.where(distances_ref < threshold)[0]
            cropped_ref = ref_points.select_by_index(cropped_indices_ref)

            if len(cropped_src.points) == 0 or len(cropped_ref.points) == 0:
                continue

            similarity = compute_similarity(cropped_src, cropped_ref)
            if similarity < best_similarity:
                best_similarity = similarity
                best_threshold = threshold
                best_neighbor = neighbor
                best_cropped_src = cropped_src
                best_cropped_ref = cropped_ref
    return best_threshold, best_neighbor, best_cropped_src, best_cropped_ref


class EpochBasedTrainer(BaseTrainer):
    def __init__(
        self,
        cfg,
        max_epoch,
        parser=None,
        cudnn_deterministic=True,
        autograd_anomaly_detection=False,
        save_all_snapshots=True,
        run_grad_check=False,
        grad_acc_steps=1,
    ):
        super().__init__(
            cfg,
            parser=parser,
            cudnn_deterministic=cudnn_deterministic,
            autograd_anomaly_detection=autograd_anomaly_detection,
            save_all_snapshots=save_all_snapshots,
            run_grad_check=run_grad_check,
            grad_acc_steps=grad_acc_steps,
        )
        self.max_epoch = max_epoch

    def before_train_step(self, epoch, iteration, data_dict) -> None:
        pass

    def before_val_step(self, epoch, iteration, data_dict) -> None:
        pass

    def after_train_step(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def before_train_epoch(self, epoch) -> None:
        pass

    def before_val_epoch(self, epoch) -> None:
        pass

    def after_train_epoch(self, epoch) -> None:
        pass

    def after_val_epoch(self, epoch) -> None:
        pass

    def train_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]:
        pass

    def val_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]:
        pass

    def after_backward(self, epoch, iteration, data_dict, output_dict, result_dict) -> None:
        pass

    def check_gradients(self, epoch, iteration, data_dict, output_dict, result_dict):
        if not self.run_grad_check:
            return
        if not self.check_invalid_gradients():
            self.logger.error('Epoch: {}, iter: {}, invalid gradients.'.format(epoch, iteration))
            torch.save(data_dict, 'data.pth')
            torch.save(self.model, 'model.pth')
            self.logger.error('Data_dict and model snapshot saved.')
            ipdb.set_trace()

    def train_epoch(self):
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.epoch)
        self.before_train_epoch(self.epoch)
        self.optimizer.zero_grad()
        total_iterations = len(self.train_loader)
        for iteration, data_dict in enumerate(self.train_loader):
            self.inner_iteration = iteration + 1
            self.iteration += 1
            data_dict = to_cuda(data_dict)
            self.before_train_step(self.epoch, self.inner_iteration, data_dict)
            self.timer.add_prepare_time()
            # forward
            output_dict, result_dict = self.train_step(self.epoch, self.inner_iteration, data_dict)

            # backward & optimization
            result_dict['loss'].backward()
            self.after_backward(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            self.check_gradients(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            self.optimizer_step(self.inner_iteration)
            # after training
            self.timer.add_process_time()
            self.after_train_step(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)

            self.summary_board.update_from_result_dict(result_dict)
            # logging
            if self.inner_iteration % self.log_steps == 0:
                summary_dict = self.summary_board.summary()
                message = get_log_string(
                    result_dict=summary_dict,
                    epoch=self.epoch,
                    max_epoch=self.max_epoch,
                    iteration=self.inner_iteration,
                    max_iteration=total_iterations,
                    lr=self.get_lr(),
                    timer=self.timer,
                )
                self.logger.info(message)
                self.write_event('train', summary_dict, self.iteration)
            torch.cuda.empty_cache()
        self.after_train_epoch(self.epoch)

        message = get_log_string(self.summary_board.summary(), epoch=self.epoch, timer=self.timer)
        self.logger.info(message)
        # scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        # snapshot
        self.save_snapshot(f'epoch-{self.epoch}.pth.tar')
        if not self.save_all_snapshots:
            last_snapshot = f'epoch-{self.epoch - 1}.pth.tar'
            if osp.exists(last_snapshot):
                os.remove(last_snapshot)

    def inference_epoch(self):
        self.set_eval_mode()
        self.before_val_epoch(self.epoch)
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.val_loader)
        pbar = tqdm.tqdm(enumerate(self.val_loader), total=total_iterations)
        estimated_transform = []
        ref_12 = []
        src_12 = []
        src_corr_12 = []
        ref_corr_12 = []
        for iteration, data_dict in pbar:
            self.inner_iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            self.before_val_step(self.epoch, self.inner_iteration, data_dict)
            timer.add_prepare_time()
            output_dict, result_dict = self.val_step(self.epoch, self.inner_iteration, data_dict)
            # print('output_dict', output_dict.keys())
            estimated_transform.append(output_dict['estimated_transform'].cpu())
            ref_12.append(output_dict['ref_points'].cpu())
            src_12.append(output_dict['src_points'].cpu())
            src_corr_12.append(output_dict['src_corr_points'].cpu())
            ref_corr_12.append(output_dict['ref_corr_points'].cpu())

            torch.cuda.synchronize()
            timer.add_process_time()
            self.after_val_step(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)

            summary_board.update_from_result_dict(result_dict)
            message = get_log_string(
                result_dict=summary_board.summary(),
                epoch=self.epoch,
                iteration=self.inner_iteration,
                max_iteration=total_iterations,
                timer=timer,
            )
            pbar.set_description(message)
            torch.cuda.empty_cache()

        ######TRE
        cases = ['02','03','04','05','06','07','08','09','10','11','12','13']

        spacing_ref = [0.8125, 0.8125, 1]
        with open(r"../../Reg/markersref.yml", "r") as f:
            file = f.read()
        content = yaml.load(file, yaml.FullLoader)
        # print(content)
        coorball = np.zeros([45, 3])
        coorclip = np.zeros([15, 3])

        databall = content['databall']
        databall_coordinatesref = np.array(databall).reshape(content['rowsball'], content['colsball'])
        for i in range(0, 45):
            coorball[i][0] = databall_coordinatesref[0][i]
            coorball[i][1] = databall_coordinatesref[1][i]
            coorball[i][2] = databall_coordinatesref[2][i]
        dataclip = content['dataclip']
        dataclip_coordinatesref = np.array(dataclip).reshape(content['rowsclip'], content['colsclip'])
        for i in range(0, 15):
            coorclip[i][0] = dataclip_coordinatesref[0][i]
            coorclip[i][1] = dataclip_coordinatesref[1][i]
            coorclip[i][2] = dataclip_coordinatesref[2][i]
        dataref = np.concatenate((coorball, coorclip), axis=0)
        TRE_list = []
        TRE_total = 0
        for case in cases:
            ##############ORG
            with open("../../Reg/markersnew" + case + ".yml", "r") as f:
                file = f.read()
            content = yaml.load(file, yaml.FullLoader)
            # print(content)
            coorball = np.zeros([45, 3])
            coorclip = np.zeros([15, 3])
            databall = content['databall']
            databall_coordinates = np.array(databall).reshape(content['rowsball'], content['colsball'])
            for i in range(0, 45):
                coorball[i][0] = databall_coordinates[0][i]
                coorball[i][1] = databall_coordinates[1][i]
                coorball[i][2] = databall_coordinates[2][i]
            # print("databall_coordinates is: \n",databall_coordinates)
            dataclip1 = content['dataclip']
            dataclip_coordinates = np.array(dataclip1).reshape(content['rowsclip'], content['colsclip'])
            for i in range(0, 15):
                coorclip[i][0] = dataclip_coordinates[0][i]
                coorclip[i][1] = dataclip_coordinates[1][i]
                coorclip[i][2] = dataclip_coordinates[2][i]
            # print("dataclip_coordinates is: \n",dataclip_coordinates)
            data = np.concatenate((coorball, coorclip), axis=0)
            # coordinate transforms
            sortdataappend = np.ones([60, 4])
            with open(r"../../Reg/Mnew" + case + ".yml", "r") as f:
                file = f.read()
            content = yaml.load(file, yaml.FullLoader)
            transform = content["data"]
            transform = np.array(transform).reshape(content['cols'], content['rows'])
            # print("gold",transform)
            for i in range(0, 60):
                sortdataappend[i] = np.append(data[i], 1)
                sortdataappend[i] = transform @ sortdataappend[i]
                data[i] = np.delete(sortdataappend[i], -1)
            # associations
            with open(r"../../Reg/associationsnew" + case + ".yml", "r") as f:
                file = f.read()
            content = yaml.load(file, yaml.FullLoader)
            sort = content["data"]
            sortdata = np.ones([60, 3])
            for i in range(0, 60):
                sortdata[i][0] = data[sort[i] - 1][0]
                sortdata[i][1] = data[sort[i] - 1][1]
                sortdata[i][2] = data[sort[i] - 1][2]
            #############################################
            # TRE
            diff = dataref - sortdata
            diff = diff * spacing_ref
            diff = torch.Tensor(diff)
            diff_clip = diff[-15:]
            TRE = diff_clip.pow(2).sum(1).sqrt()
            TRE = TRE.mean()

            ############################## pre
            ##################1#########################
            with open("../../Reg/markersnew" + case + ".yml", "r") as f:
                file = f.read()
            content = yaml.load(file, yaml.FullLoader)
            # print(content)
            coorball = np.zeros([45, 3])
            coorclip = np.zeros([15, 3])
            databall = content['databall']
            databall_coordinates = np.array(databall).reshape(content['rowsball'], content['colsball'])
            for i in range(0, 45):
                coorball[i][0] = databall_coordinates[0][i]
                coorball[i][1] = databall_coordinates[1][i]
                coorball[i][2] = databall_coordinates[2][i]
            # print("databall_coordinates is: \n",databall_coordinates)
            dataclip1 = content['dataclip']
            dataclip_coordinates = np.array(dataclip1).reshape(content['rowsclip'], content['colsclip'])
            for i in range(0, 15):
                coorclip[i][0] = dataclip_coordinates[0][i]
                coorclip[i][1] = dataclip_coordinates[1][i]
                coorclip[i][2] = dataclip_coordinates[2][i]
            # print("dataclip_coordinates is: \n",dataclip_coordinates)
            data = np.concatenate((coorball, coorclip), axis=0)
            # coordinate transforms
            sortdataappend = np.ones([60, 4])
            # with open("../../Reg/Mpredict" + case + ".yml", "r") as f:
            #     file = f.read()
            # content = yaml.load(file, yaml.FullLoader)
            transform = np.array(estimated_transform[int(case)-2])
            transform[:, 3] *= 100
            for i in range(0, 60):
                sortdataappend[i] = np.append(data[i], 1)
                sortdataappend[i] = transform @ sortdataappend[i]
                data[i] = np.delete(sortdataappend[i], -1)

            # associations
            with open(r"../../Reg/associationsnew" + case + ".yml", "r") as f:
                file = f.read()
            content = yaml.load(file, yaml.FullLoader)
            sort = content["data"]
            sortdata = np.ones([60, 3])
            for i in range(0, 60):
                sortdata[i][0] = data[sort[i] - 1][0]
                sortdata[i][1] = data[sort[i] - 1][1]
                sortdata[i][2] = data[sort[i] - 1][2]

            diff = dataref - sortdata
            diff = torch.Tensor(diff)
            diff_clip = diff[-15:]
            TREpre = diff_clip.pow(2).sum(1).sqrt()
            TREpre = TREpre.mean()

            diff_sum = abs(TRE - TREpre)
            TRE_list.append(round(diff_sum.item(),5))
            TRE_total += round(diff_sum.item(),5)
            # print(case,"TRE", diff_sum.item(), "mm")

            ref_points = ref_12[int(case)-2].numpy()
            ref = o3d.geometry.PointCloud()
            ref.points = o3d.utility.Vector3dVector(ref_points)
            o3d.io.write_point_cloud("rigidResults/ref"+case+".ply", ref)

            ref_corr_points = ref_corr_12[int(case) - 2].numpy()
            ref_corr = o3d.geometry.PointCloud()
            ref_corr.points = o3d.utility.Vector3dVector(ref_corr_points)
            o3d.io.write_point_cloud("rigidResults/ref_corr" + case + ".ply", ref_corr)

            src_points = src_12[int(case)-2].numpy()
            src = o3d.geometry.PointCloud()
            src.points = o3d.utility.Vector3dVector(src_points)
            o3d.io.write_point_cloud("rigidResults/src"+case+".ply", src)

            src_corr_points = src_corr_12[int(case)-2].numpy()
            src_corr = o3d.geometry.PointCloud()
            src_corr.points = o3d.utility.Vector3dVector(src_corr_points)
            o3d.io.write_point_cloud("rigidResults/src_corr"+case+".ply", src_corr)
        print('mean',round(TRE_total/12,5),'mm', "TRE", TRE_list)
        for i, tensor in enumerate(estimated_transform,start=2):
            formatted_matrix = np.array2string(tensor.numpy(), formatter={'float_kind': lambda x: f"{x:.5f}"})
            file_path = f"rigidResults/estimated_transform{i:02d}.txt"
            with open(file_path, 'w') as file:
                file.write(formatted_matrix)
        #################
        self.after_val_epoch(self.epoch)
        summary_dict = summary_board.summary()

        message = '[Val] ' + get_log_string(summary_dict, epoch=self.epoch, timer=timer)
        self.logger.critical(message)
        self.write_event('val', summary_dict, self.epoch)
        self.set_train_mode()

    def Deform(self):
        cases = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']

        neighbor_range = range(50, 1000, 50)
        threshold_range = np.arange(0.05, 0.5, 0.05)

        for case in cases:
            with open("rigidResults/estimated_transform" + case + ".txt", 'r') as file:
                content = file.read()
            estimated_transform = np.fromstring(content.replace('[', '').replace(']', '').replace('\n', ' '),
                                                sep=' ')
            estimated_transform = estimated_transform.reshape((4, 4))

            src_corr_points = o3d.io.read_point_cloud(r"rigidResults/src_corr" + case + ".ply")
            src_corr_points = src_corr_points.transform(estimated_transform)
            src_points = o3d.io.read_point_cloud(r"rigidResults/src" + case + ".ply")
            src_points = src_points.transform(estimated_transform)

            ref_corr_points = o3d.io.read_point_cloud(r"rigidResults/ref_corr" + case + ".ply")
            ref_points = o3d.io.read_point_cloud(r"rigidResults/ref" + case + ".ply")

            best_threshold, best_neighbor, cropped_src, cropped_ref = auto_tune_parameters(
                src_points, src_corr_points, ref_points, ref_corr_points, neighbor_range, threshold_range)

            print(f"[Case {case}] Best Threshold: {best_threshold:.3f}, Best Neighbor: {best_neighbor}")

            o3d.io.write_point_cloud(r"rigidResults/cropped_src_points" + case + ".ply", cropped_src)
            o3d.io.write_point_cloud(r"rigidResults/cropped_ref_points" + case + ".ply", cropped_ref)
            # ==============================================

        chamfer_values = []
        chamfer_org_values = []
        ratio_values = []
        diff, time0, ratiototal = 0, 0, 0
        chamfer_dist_org = 0
        chamfer_dist = 0
        ratio = 0

        for case in cases:

            ################# 形变
            config = {
                "gpu_mode": True,

                "iters": 10,
                "lr": 0.1,
                "max_break_count": 1,
                "break_threshold_ratio": 0.015,


                "motion_type": "Sim3",
                "rotation_format": "euler",

                "m": 5,
                "k0": -1,
                "depth": 1,
                "width": 8,
                "act_fn": "relu",

                "w_reg": 0,
                "w_ldmk": 0,
                "w_cd": 0.1
            }

            config = edict(config)
            config.device = torch.cuda.current_device()

            S = r"rigidResults/cropped_src_points"+case+".ply"
            T = r"rigidResults/cropped_ref_points" + case + ".ply"

            """read S, sample pts"""
            warped = o3d.io.read_point_cloud(T)
            pcd1 = o3d.io.read_point_cloud(T)
            pcd1.paint_uniform_color([0, 0.706, 1])
            tgt_pcd = np.asarray(pcd1.points, dtype=np.float32)

            """read T, sample pts"""
            pcd2 = o3d.io.read_point_cloud(S)
            src_pcd = np.asarray(pcd2.points, dtype=np.float32)


            """load data"""
            tgt_pcd, src_pcd = map(lambda x: torch.from_numpy(x).to(config.device), [tgt_pcd, src_pcd])
            mesh_vert = torch.from_numpy(np.asarray(src_pcd.cpu(), dtype=np.float32)).to(config.device)
            """construct model"""
            NDP = Deformation_Pyramid(depth=config.depth,
                                      width=config.width,
                                      device=config.device,
                                      k0=config.k0,
                                      m=config.m,
                                      nonrigidity_est=config.w_reg > 0,
                                      rotation_format=config.rotation_format,
                                      motion=config.motion_type)

            """cancel global translation"""
            s_sample = src_pcd
            t_sample = tgt_pcd
            for level in range(NDP.n_hierarchy):

                """freeze non-optimized level"""
                NDP.gradient_setup(optimized_level=level)

                optimizer = optim.Adam(NDP.pyramid[level].parameters(), lr=config.lr)

                break_counter = 0
                loss_prev = 1e+6

                """optimize current level"""
                for iter in range(config.iters):

                    s_sample_warped, data = NDP.warp(s_sample, max_level=level, min_level=level)

                    loss = compute_truncated_chamfer_distance(s_sample_warped[None], t_sample[None], trunc=1e+9)

                    if level > 0 and config.w_reg > 0:
                        nonrigidity = data[level][1]
                        target = torch.zeros_like(nonrigidity)
                        reg_loss = torch.nn.BCELoss(nonrigidity, target)
                        loss = loss + config.w_reg * reg_loss

                    # early stop
                    if loss.item() < 1e-4:
                        break
                    if abs(loss_prev - loss.item()) < loss_prev * config.break_threshold_ratio:
                        break_counter += 1
                    if break_counter >= config.max_break_count:
                        break
                    loss_prev = loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # use warped points for next level
                s_sample = s_sample_warped.detach()

            """warp-original mesh verttices"""
            NDP.gradient_setup(optimized_level=-1)

            warped_vert, data = NDP.warp(mesh_vert)

            warped_vert = warped_vert.detach().cpu().numpy()
            warped.points = o3d.utility.Vector3dVector(warped_vert)

            o3d.io.write_point_cloud("nonrigidResults/val-prem/warped" + case + ".ply", warped)
            ref = pcd1
            src = pcd2

            chamfer_dist_org = oneway_chamfer_distance(pcd2, pcd1)
            chamfer_dist = oneway_chamfer_distance(warped, pcd1)
            ratio = AreaChange(src, warped)
        chamfer_values.append(chamfer_dist)
        chamfer_org_values.append(chamfer_dist_org)
        ratio_values.append(ratio)


        mean_chamfer = np.mean(chamfer_values)
        mean_chamfer_org = np.mean(chamfer_org_values)
        mean_ratio = np.mean(ratio_values)

        # Calculate variances
        var_chamfer = np.var(chamfer_values)
        var_chamfer_org = np.var(chamfer_org_values)
        var_ratio = np.var(ratio_values)

        # # Print results
        print(f"ChamferORG Mean: {mean_chamfer_org * 100:.2f}±{var_chamfer_org * 100:.2f}")
        print(f"Chamfer Mean: {mean_chamfer * 100:.2f}±{var_chamfer * 100:.2f}")
        print(f"Area Change Mean: {mean_ratio * 100:.2f}±{var_ratio * 100:.2f}")
    def run(self):
        assert self.train_loader is not None
        assert self.val_loader is not None

        if self.args.resume:
            self.load_snapshot(osp.join(self.snapshot_dir, 'snapshot.pth.tar'))
        elif self.args.snapshot is not None:
            self.load_snapshot(self.args.snapshot)
        self.set_train_mode()

        while self.epoch < self.max_epoch:
            self.epoch += 1
            self.train_epoch()
            self.inference_epoch()
            self.Deform()


def oneway_chamfer_distance(pcd1, pcd2):
    dist1 = pcd1.compute_point_cloud_distance(pcd2)
    # dist2 = pcd2.compute_point_cloud_distance(pcd1)
    # chamfer_dist = (np.mean(dist1) + np.mean(dist2)) / 2
    return np.mean(dist1)

def AreaChange(src, warped):
    alpha = 0.5
    mesh1 = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(src, alpha)
    surface_area1 = mesh1.get_surface_area()
    mesh2 = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(warped, alpha)
    surface_area2 = mesh2.get_surface_area()
    ratio = abs(surface_area1 - surface_area2) / surface_area1
    return ratio
