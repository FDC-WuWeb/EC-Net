import torch
import torch.nn as nn
from .rigid_body import _6d_to_SO3, euler_to_SO3, quaternion_to_SO3, exp_se3, exp_so3, _copysign
# from .position_encoding import *
import torch.nn.functional as F
from torch.autograd.functional import jacobian
import math

class Deformation_KAN():

    def __init__(self, depth, width, device, k0, m, rotation_format, nonrigidity_est=False, motion='SE3'):

        pyramid = []

        assert motion in [ "Sim3", "SE3", "sflow"]

        for i in range (m):
            pyramid.append(
                DKLayer(depth,
                         width,
                         k0,
                         i+1,
                         rotation_format,
                         nonrigidity_est=nonrigidity_est & (i!=0),
                         motion=motion
                         ).to(device)
            )


        self.pyramid = pyramid
        self.n_hierarchy = m

    def warp(self, x, max_level=None, min_level=0):

        if max_level is None:
            max_level = self.n_hierarchy - 1

        assert max_level < self.n_hierarchy, "more level than defined"

        data = {}

        for i in range(min_level, max_level + 1):
            x, nonrigidity = self.pyramid[i](x)
            data[i] = (x, nonrigidity)


        return x, data

    def gradient_setup(self, optimized_level):

        assert optimized_level < self.n_hierarchy, "more level than defined"

        # optimize current level, freeze the other levels
        for i in range( self.n_hierarchy):
            net = self.pyramid[i]
            if i == optimized_level:
                for param in net.parameters():
                    param.requires_grad = True
            else:
                for param in net.parameters():
                    param.requires_grad = False



class DKLayer(nn.Module):
    def __init__(self, depth, width, k0, m, rotation_format="euler", nonrigidity_est=False, motion='SE3'):
        super().__init__()

        self.k0 = k0
        self.m = m
        dim_x = 6
        self.nonrigidity_est = nonrigidity_est
        self.motion = motion
        # self.input= nn.Sequential(nn.Linear(dim_x,width), nn.ReLU())
        self.input= nn.Sequential(KANLinear(dim_x,width))
        self.kan = KAN(depth=depth,width=width)
        # self.kan = KAN(depth=depth,width=width)

        self.rotation_format = rotation_format

        """rotation branch"""
        if self.motion in [ "Sim3", "SE3"] :

            if self.rotation_format in [ "axis_angle", "euler" ]:
                # self.rot_brach = nn.Linear(width, 3)
                self.rot_brach = KANLinear(width, 3)
            elif self.rotation_format == "quaternion":
                # self.rot_brach = nn.Linear(width, 4)
                self.rot_brach = KANLinear(width, 4)
            elif self.rotation_format == "6D":
                # self.rot_brach = nn.Linear(width, 6)
                self.rot_brach = KANLinear(width, 6)

            if self.motion == "Sim3":
                # self.s_branch = nn.Linear(width, 1) # scale branch
                self.s_branch = KANLinear(width, 1) # scale branch

        """translation branch"""
        # self.trn_branch = nn.Linear(width, 3)
        self.trn_branch = KANLinear(width, 3)

        """rigidity branch"""
        if self.nonrigidity_est:
            # self.nr_branch = nn.Linear(width, 1)
            self.nr_branch = KANLinear(width, 1)
            self.sigmoid = nn.Sigmoid()

        # Apply small scaling on the MLP output, s.t. the optimization can start from near identity pose
        self.mlp_scale = 0.001

        self._reset_parameters()

    def forward (self, x):

        fea = self.posenc( x )
        fea = self.input(fea)
        fea = self.kan(fea)

        t = self.mlp_scale * self.trn_branch ( fea )

        if self.motion == "SE3":
            R = self.get_Rotation(fea)
            x_ = (R @ x[..., None]).squeeze() + t

        elif self.motion == "Sim3":
            R = self.get_Rotation(fea)
            s = self.mlp_scale * self.s_branch(fea) + 1  # optimization starts with scale==1
            x_ = s * (R @ x[..., None]).squeeze() + t

        else: # scene flow
            x_ = x + t

        if self.nonrigidity_est:
            nonrigidity =self.sigmoid( self.mlp_scale * self.nr_branch(fea) )
            x_ = x + nonrigidity * (x_ - x)
            nonrigidity = nonrigidity.squeeze()
        else:
            nonrigidity = None

        return x_.squeeze(), nonrigidity

    def get_Rotation (self, fea):

        R = self.mlp_scale * self.rot_brach( fea )

        if self.rotation_format == "euler":
            R = euler_to_SO3(R)
        elif self.rotation_format == "axis_angle":
            theta = torch.norm(R, dim=-1, keepdim=True)
            w = R / theta
            R = exp_so3(w, theta)
        elif self.rotation_format =='quaternion':
            s = (R * R).sum(1)
            R = R / _copysign(torch.sqrt(s), R[:, 0])[:, None]
            R = quaternion_to_SO3(R)
        elif self.rotation_format == "6D":
            R = _6d_to_SO3(R)

        return R


    def posenc(self, pos):
        pi = 3.14
        x_position, y_position, z_position = pos[..., 0:1], pos[..., 1:2], pos[..., 2:3]
        # mul_term = ( 2 ** (torch.arange(self.m, device=pos.device).float() + self.k0) * pi ).reshape(1, -1)
        mul_term = (2 ** (self.m + self.k0)  )#.reshape(1, -1)

        sinx = torch.sin(x_position * mul_term)
        cosx = torch.cos(x_position * mul_term)
        siny = torch.sin(y_position * mul_term)
        cosy = torch.cos(y_position * mul_term)
        sinz = torch.sin(z_position * mul_term)
        cosz = torch.cos(z_position * mul_term)
        pe = torch.cat([sinx, cosx, siny, cosy, sinz, cosz], dim=-1)
        return pe


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



class Nerfies_Deformation(nn.Module):
    '''
    Our re-implementation of the deformation model from [Nerfies, ICCV'21], https://arxiv.org/abs/2011.12948
    '''
    def __init__(self, depth=7, width=128, max_iter = 5000):
        super().__init__()

        self.k0 = -3
        self.m=6
        dim_x = self.m * 6 + 3
        self.input= nn.Sequential( nn.Linear(dim_x,width), nn.ReLU())
        self.mlp = MLP(depth=depth,width=width)
        self.w_branch = nn.Linear(width, 3)
        self.v_branch = nn.Linear(width, 3)

        self.max_iter = max_iter
        self.N = 0.6 * max_iter

    def forward(self, x, iter ):
        warpped_x = self.warp(x, iter)
        J = self.batched_jacobian(self.warp, x, iter)
        return warpped_x, J


    def batched_jacobian(self, f, x, iter):
        f_sum = lambda x: torch.sum(f(x, iter), axis=0)
        return jacobian(f_sum, x).transpose(0,1)


    def posenc(self, pos, iter):

        pi = 3.14

        # sliding window
        a = self.m * iter / self.N
        w_a = ( 1 - torch.cos( torch.clamp(a-torch.arange(self.m, device=pos.device).float(), min=0, max=1) * pi ) ) / 2
        w_a = w_a[None]

        x_position, y_position, z_position = pos[..., 0:1], pos[..., 1:2], pos[..., 2:3]
        mul_term = (
                2 ** (torch.arange(self.m, device=pos.device).float() + self.k0) * pi
                ).reshape(1, -1)

        sinx = torch.sin(x_position * mul_term) * w_a
        cosx = torch.cos(x_position * mul_term) * w_a
        siny = torch.sin(y_position * mul_term) * w_a
        cosy = torch.cos(y_position * mul_term) * w_a
        sinz = torch.sin(z_position * mul_term) * w_a
        cosz = torch.cos(z_position * mul_term) * w_a
        position_code = torch.cat([sinx, cosx, siny, cosy, sinz, cosz], dim=-1)
        position_code = torch.cat( [pos, position_code], dim= -1)
        return position_code

    def warp (self, x, iter) :
        fea = self.posenc(x, iter)
        fea = self.input(fea)
        fea = self.mlp(fea)
        w = self.w_branch(fea)
        v = self.v_branch(fea)
        theta = torch.norm(w, dim=-1, keepdim=True)
        w = w/theta
        v = v/theta
        R, t = exp_se3(w, v, theta)
        _x = ( R @ x[..., None] + t ).squeeze()
        return _x.squeeze()


class Neural_Prior(torch.nn.Module):
    '''
    Borrow from [Neural Scene flow Prior, NIPS'21], https://arxiv.org/abs/2111.01253
    '''
    def __init__(self, dim_x=3, filter_size=128, act_fn='relu'):
        super().__init__()
        # input layer (default: xyz -> 128)
        self.layer1 = torch.nn.Linear(dim_x, filter_size)
        # hidden layers (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        self.layer3 = torch.nn.Linear(filter_size, filter_size)
        self.layer4 = torch.nn.Linear(filter_size, filter_size)
        self.layer5 = torch.nn.Linear(filter_size, filter_size)
        self.layer6 = torch.nn.Linear(filter_size, filter_size)
        self.layer7 = torch.nn.Linear(filter_size, filter_size)
        self.layer8 = torch.nn.Linear(filter_size, filter_size)
        # output layer (default: 128 -> 3)
        self.layer9 = torch.nn.Linear(filter_size, 3)

        # activation functions
        if act_fn == 'relu':
            self.act_fn = torch.nn.functional.relu
        elif act_fn == 'sigmoid':
            self.act_fn = torch.nn.functional.sigmoid

    def forward(self, x):
        x = self.act_fn(self.layer1(x))
        x = self.act_fn(self.layer2(x))
        x = self.act_fn(self.layer3(x))
        x = self.act_fn(self.layer4(x))
        x = self.act_fn(self.layer5(x))
        x = self.act_fn(self.layer6(x))
        x = self.act_fn(self.layer7(x))
        x = self.act_fn(self.layer8(x))
        x = self.layer9(x)

        return x


class KAN(torch.nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        # self.pts_linears = nn.ModuleList( [nn.Linear(width, width) for i in range(depth - 1)])
        self.pts_linears = nn.ModuleList([KANLinear(width, width) for i in range(depth - 1)])

    def forward(self, x):
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
            # x = F.relu(x)
        return x

class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
                .expand(in_features, -1)
                .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()


    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )


    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )

# class KAN(torch.nn.Module):
#     def __init__(
#         self,
#         layers_hidden,
#         grid_size=5,
#         spline_order=3,
#         scale_noise=0.1,
#         scale_base=1.0,
#         scale_spline=1.0,
#         base_activation=torch.nn.SiLU,
#         grid_eps=0.02,
#         grid_range=[-1, 1],
#     ):
#         super(KAN, self).__init__()
#         self.grid_size = grid_size
#         self.spline_order = spline_order
#
#         self.layers = torch.nn.ModuleList()
#         for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
#             self.layers.append(
#                 KANLinear(
#                     in_features,
#                     out_features,
#                     grid_size=grid_size,
#                     spline_order=spline_order,
#                     scale_noise=scale_noise,
#                     scale_base=scale_base,
#                     scale_spline=scale_spline,
#                     base_activation=base_activation,
#                     grid_eps=grid_eps,
#                     grid_range=grid_range,
#                 )
#             )
#
#     def forward(self, x: torch.Tensor, update_grid=False):
#         for layer in self.layers:
#             if update_grid:
#                 layer.update_grid(x)
#             x = layer(x)
#         return x
#
#     def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
#         return sum(
#             layer.regularization_loss(regularize_activation, regularize_entropy)
#             for layer in self.layers
#         )
