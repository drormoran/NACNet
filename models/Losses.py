import kornia.geometry
from torch import nn
from utils import TensorSet
from abc import ABC, abstractmethod
import torch
from Geometry import stereo_2d


class RegLoss(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch, supp_pred):
        pass


class MSELoss(RegLoss):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, batch, supp_pred):
        gt_shape, _, _, _ = batch
        return self.mse_loss(supp_pred["pred_shape"], gt_shape)


class EpiSymLoss(RegLoss):
    def __init__(self, args):
        super().__init__()
        self.geo_loss_margin = args.geo_loss_margin
        if args.loss_type == "SED":
            self.sym_func = kornia.geometry.symmetrical_epipolar_distance
        elif args.loss_type == "Sampson":
            self.sym_func = kornia.geometry.sampson_epipolar_distance
        else:
            raise NotImplementedError

    def forward(self, batch, supp_pred):
        _, _, _, supp_data = batch
        pts_virt = supp_data['pts_virt']
        pts1_virts, pts2_virts = pts_virt[:, :, :2], pts_virt[:, :, 2:]

        pred_mat = supp_pred["pred_shape"].reshape(-1, 3, 3)
        geod = self.sym_func(pts1_virts, pts2_virts, pred_mat)
        essential_loss = torch.min(geod, self.geo_loss_margin * torch.ones_like(geod))
        return essential_loss.mean(dim=1).mean()


class ReprojError(RegLoss):
    def __init__(self, args):
        super().__init__()
        self.gradient_reduction = args.gradient_reduction
        self.infinity_pts_margin = 1e-4
        self.hinge_loss_weight = 1
        self.detach_crct_pts = args.detach_crct_pts

    def forward(self, batch, supp_pred):
        # Get points
        _, _, gt_outliers, _ = batch
        sampled_pts = supp_pred["corrected_pts"]
        if self.detach_crct_pts:
            sampled_pts = sampled_pts.detach()

        gt_outliers = gt_outliers
        gt_inliers = gt_outliers.apply_func(lambda x: 1 - x)
        inliers_pts = sampled_pts[gt_inliers]
        inliers_pred_3dpts = supp_pred["pred_features"][gt_inliers]
        inliers_pts1, inliers_pts2 = TensorSet.matches_to_points(inliers_pts)

        # Get cameras
        pred_q = supp_pred['pred_q']
        pred_R = stereo_2d.quaternion_to_rotation_mat(pred_q)
        pred_t = supp_pred['pred_t']
        pred_P1 = torch.cat((torch.eye(3), torch.zeros(3, 1)), dim=-1).expand(pred_R.shape[0], -1, -1).to(pred_R.device)
        pred_P2 = torch.cat((pred_R, pred_t.unsqueeze(-1)), dim=-1)

        repr_err1 = self.calc_reprojection_err(inliers_pred_3dpts, pred_P1, inliers_pts1)
        repr_err2 = self.calc_reprojection_err(inliers_pred_3dpts, pred_P2, inliers_pts2)

        return (repr_err1.mean(dim=1) + repr_err2.mean(dim=1)).mean()

    def calc_reprojection_err(self, pts3d, Pmat, pts2d):
        repr_pts2d, depth = TensorSet.project_3d_pts(pts3d, Pmat, self.gradient_reduction)

        # Calculate reprojection error and hinge loss
        rep_err = ((repr_pts2d - pts2d) ** 2).sum(dim=2).sqrt()
        hinge_loss = (depth - self.infinity_pts_margin) * (-1) * self.hinge_loss_weight
        projected_pts_mask = (depth > self.infinity_pts_margin)
        err = TensorSet.where(projected_pts_mask, rep_err, hinge_loss)
        return err


class ClassficationLoss(nn.Module):
    def __init__(self, args):
        super(ClassficationLoss, self).__init__()
        self.inliers_cls_loss_weight = args.inliers_cls_loss_weight

    def forward(self, supp_pred, batch):
        _, _, gt_outliers, _ = batch
        pred_outliers = supp_pred["pred_outliers"]
        err = TensorSet.binary_cross_entropy(pred_outliers, gt_outliers)
        outliers_err = err.masked_mean(gt_outliers)
        inliers_err = err.masked_mean(gt_outliers.apply_func(lambda x: (x == 0).float()))
        batch_err = outliers_err + inliers_err * self.inliers_cls_loss_weight
        return batch_err.mean()


class StereoNoiseLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sqrt_noise_loss = args.sqrt_noise_loss
        self.noise_func = stereo_2d.StereoNoiseFunc(args.noise_calc_func)

    def forward(self, supp_pred, batch):
        noise_err_dict = self.noise_func(supp_pred['corrected_pts'], batch, self.sqrt_noise_loss)
        noise_err = noise_err_dict["noise_err1"] + noise_err_dict["noise_err2"]
        return noise_err.mean()





