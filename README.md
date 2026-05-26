# EC-Net: Edge-guided and Correspondence-constrained Network for Laparoscopic Low-Overlap Point Cloud Registration (ESWA-D-26-10668)
<div align="center">
  <img src="/Fig1.svg">
</div>
This repository provides the implementation of EC-Net, an end-to-end deep learning framework designed to registering organ surface point clouds reconstructed separately from preoperative CT and intraoperative video for Laparoscopic Augmented Reality.

Due to the narrow field of view and large organ deformations, exsiting registration methods often fails. EC-Net solves this by proposing:

🎯 Edge-Guidance Mechanism: Prevents rigid registration from falling into local optima by selecting edge points to guide regsitration in texture-less regions.

🔗 Correspondence-Constraint Strategy: Restricts disordered spatial mapping during complex non-rigid registration.

⚡ KAN Architecture: Utilizes the Kolmogorov-Arnold Network to capture large deformations with highly efficient parameters.

Performance: Compared with SOTA methods, EC-Net significantly boosts accuracy (up to +36%) and speed (up to +20%), while demonstrating excellent generalization and noise robustness for clinical deployment.
