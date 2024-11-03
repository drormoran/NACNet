import numpy as np
import h5py
import torch
from Geometry import stereo_2d
from Datasets import BaseDataset
from plots import plot_stereo_2d
from utils import TensorSet
from models import Losses
from abc import ABC, abstractmethod
from PIL import Image
import os


class BaseStereoDataset(BaseDataset.BaseDataset, ABC):
    def __init__(self, data_type, args):
        super(BaseStereoDataset, self).__init__(data_type, args)
        self.noise_free = args.noise_free

    @abstractmethod
    def get_stereo_data(self, idx):
        pass

    def get_item(self, idx):
        xs, R, t, outliers_mask, snn_ratio, e_gt = self.get_stereo_data(idx)

        # Get corrected pts
        pts1, pts2 = np.split(xs, 2, axis=-1)
        crct_pts1, crct_pts2 = stereo_2d.correct_matches(e_gt, pts1, pts2)
        crct_pts = np.concatenate([crct_pts1, crct_pts2], axis=1).astype('float64')

        # Remove noise
        if self.noise_free:
            inliers_mask = (outliers_mask == 0).squeeze(1)
            if inliers_mask.any():
                xs[inliers_mask] = crct_pts[inliers_mask]

        # Get virtual points
        pts1_virt, pts2_virt = self.get_grid_correspondences(e_gt)
        pts_virt = np.concatenate([pts1_virt, pts2_virt], axis=1).astype('float64')

        gt_shape = stereo_2d.E_to_shape(e_gt)

        supp_data = dict(snn_ratio=snn_ratio, e_gt=e_gt, pts_virt=pts_virt, R=R, t=t, idx=idx, crct_pts=crct_pts)
        return self.to_torch(gt_shape, xs, outliers_mask, supp_data)

    @ classmethod
    def collate_sets(cls, batch):
        supp_data_collate_func = dict(snn_ratio=TensorSet.TensorSet, e_gt=torch.stack,  pts_virt=torch.stack,
                                      R=torch.stack, t=torch.stack, idx=torch.tensor, crct_pts=TensorSet.TensorSet)
        return TensorSet.collate_sets(batch, supp_data_collate_func)

    @staticmethod
    def get_input_dim():
        return 4

    @staticmethod
    def get_output_dim():
        return 7

    @classmethod
    def get_eval_regression_func(cls, args):
        return stereo_2d.eval_essential_mat

    @classmethod
    def get_eval_noise_func(cls, args):
        return stereo_2d.StereoNoiseFunc("triangulated_euc_dist")

    @classmethod
    def get_plot_func(cls, args):
        return plot_stereo_2d.plot_essential_mat_transformation

    @classmethod
    def get_normalization_params(cls, args):
        return cls.dummy_normalization()

    @classmethod
    def get_strict_estimator(cls, args):
        return stereo_2d.EssentialMatEstimator(args)

    @classmethod
    def prepare_model_input(cls, args, x, supp_data):
        resotre_args = {}

        # Filter by snn
        snn_mask = supp_data['snn_ratio'] <= torch.tensor([args.snn_threshold]).reshape((1, 1, 1)).to(supp_data['snn_ratio'].device)
        resotre_args['snn_mask'] = snn_mask
        x = x[snn_mask]

        return x, resotre_args

    @classmethod
    def restore_model_output(cls, args, pred_shape, pred_outliers, resotre_args):
        all_pred_outliers = resotre_args['snn_mask'].astype(torch.float).apply_func(lambda x: 1 - x)
        all_pred_outliers[resotre_args['snn_mask']] = pred_outliers

        assert pred_shape.shape[-1] == 7
        pred_q, pred_t = pred_shape[:, :4], pred_shape[:, 4:]
        supp_pred = dict(pred_q=pred_q, pred_t=pred_t)

        if args.pred_from_samples:
            crct_pts = resotre_args['corrected_pts'].detach() if args.detach_crct_pts else resotre_args['corrected_pts']
            pred_shape = stereo_2d.weighted_8points(crct_pts, resotre_args['pred_logits'])
        else:
            pred_shape = stereo_2d.q_t_to_shape(pred_q, pred_t)

        return pred_shape, all_pred_outliers, resotre_args['corrected_pts'], supp_pred

    def get_grid_correspondences(self, e_gt):
        step = 0.1
        xx, yy = np.meshgrid(np.arange(-1, 1, step), np.arange(-1, 1, step))
        # Points in first image before projection
        pts1_virt_b = np.float32(np.vstack((xx.flatten(), yy.flatten())).T)
        # Points in second image before projection
        pts2_virt_b = np.float32(pts1_virt_b)

        return stereo_2d.correct_matches(e_gt.reshape(3,3), pts1_virt_b, pts2_virt_b)

    @classmethod
    def get_global_eval_func(cls, args):
        return stereo_2d.EvalGlobalErr(args)

    @classmethod
    def get_regression_loss(cls, args):
        if args.loss_type == "MSE":
            return Losses.MSELoss()
        if args.loss_type == "Proj":
            return Losses.ReprojError(args)
        else:
            return Losses.EpiSymLoss(args)

    @classmethod
    def get_noise_loss(cls, args):
        return Losses.StereoNoiseLoss(args)

    @classmethod
    def get_features_head_dims(cls, args):
        return 3

    @classmethod
    def get_meta_data(cls, args, batch):
        meta_dict = super(BaseStereoDataset, cls).get_meta_data(args, batch)
        _, _, _, supp_data = batch
        meta_dict["meta_idx"] = supp_data["idx"].cpu()
        return meta_dict

    @classmethod
    def get_noise_dist(cls, args, batch):
        gt_shape, sampled_pts, gt_outliers, supp_data = batch
        es_gt = stereo_2d.shape_to_E(gt_shape).cpu().numpy()

        noise_mean = []
        noise_std = []
        noise_samples = []
        for i in range(sampled_pts.batch_size):
            in_pts_mask = gt_outliers[i].squeeze(1) == 0
            in_xs = sampled_pts[i][in_pts_mask].cpu().numpy()
            in_pts1, in_pts2 = np.split(in_xs, 2, axis=-1)
            crct_pts1, crct_pts2 = stereo_2d.correct_matches(es_gt[i], in_pts1, in_pts2)

            noise_samples_i = np.concatenate((crct_pts1 - in_pts1, crct_pts2 - in_pts2), axis=-1)
            noise_samples.append(torch.from_numpy(noise_samples_i))

            in_rep_err, _, _ = stereo_2d.reprojection_err(in_pts1, crct_pts1, in_pts2, crct_pts2)
            noise_mean.append(in_rep_err.mean())
            noise_std.append(in_rep_err.std())

        return dict(noise_mean=torch.tensor(noise_mean), noise_std=torch.tensor(noise_std), noise_samples=TensorSet.TensorSet(noise_samples))

    @classmethod
    def get_plot_order_dict(cls, args, err_dict):
        main_err = torch.maximum(err_dict['err_q'], err_dict['err_t'])
        plot_order_dict = dict(base=torch.randperm(main_err.shape[0]))
        plot_order_dict['best'] = main_err.argsort(descending=False)
        plot_order_dict['worst'] = main_err.argsort(descending=True)
        plot_order_dict['failure'] = (main_err > 5).nonzero().squeeze(1)

        return main_err, plot_order_dict


class RealStereoDataset(BaseStereoDataset, ABC):
    def __init__(self, data_type, args):
        super(RealStereoDataset, self).__init__(data_type, args)
        self.stereo_geod_th = args.stereo_geod_th
        self.stereo_geod_method = args.stereo_geod_method
        self.data_file_path = self.get_data_file_path(args.data_path, self.data_type, args.desc_name)
        self.data = None
        self.data_len = self.get_data_len(args.data_path, self.data_type, args.desc_name)

    def __len__(self):
        return self.data_len

    def __del__(self):
        if self.data is not None:
            self.data.close()

    @classmethod
    def get_outliers_rate(cls, args):
        return 0.99  # TODO make it more accurate

    @classmethod
    def get_noise_norm_func(cls, args):
        if args.noise_norm_func == "max_noise":
            max_noise = args.stereo_geod_th
            return lambda x: x * max_noise

        elif args.noise_norm_func == "from_data":
            noise_mean, noise_std = cls.get_noise_mean_n_std(args)
            return NoiseNormalization(noise_mean, noise_std)

        else:
            raise NotImplementedError

    @classmethod
    def get_noise_mean_n_std(cls, args):
        raise NotImplementedError

    def get_stereo_data(self, idx):
        if self.data is None:
            self.data = h5py.File(self.data_file_path, 'r')

        xs = np.asarray(self.data['xs'][str(idx)]).squeeze(0)
        snn_ratio = np.asarray(self.data['ratios'][str(idx)]).reshape(-1, 1)
        R = np.asarray(self.data['Rs'][str(idx)])
        t = np.asarray(self.data['ts'][str(idx)])
        e_gt = stereo_2d.calc_essential_mat(t, R)

        # Get outliers mask
        if self.stereo_geod_method in ["RepErr", "RepErr12"]:
            rep_err, rep_err1, rep_err2, depth1, depth2 = stereo_2d.calc_reprojection_error(xs, e_gt, R, t)

            # Get outliers mask
            outliers_mask = np.logical_or(depth1 < 0, depth2 < 0)
            outliers_mask = np.logical_or(outliers_mask, np.isnan(depth1))
            if self.stereo_geod_method == "RepErr":
                outliers_mask = np.logical_or(outliers_mask, rep_err > self.stereo_geod_th)
            elif self.stereo_geod_method == "RepErr12":
                outliers_mask = np.logical_or(outliers_mask, rep_err1 > self.stereo_geod_th)
                outliers_mask = np.logical_or(outliers_mask, rep_err2 > self.stereo_geod_th)
            else:
                raise NotImplementedError

            outliers_mask = outliers_mask.astype(float).reshape(-1, 1)

        elif self.stereo_geod_method == "Sampson":
            raise NotImplementedError

        elif self.stereo_geod_method == "SED":
            geod_dist = np.asarray(self.data['ys'][str(idx)])
            outliers_mask = (geod_dist > self.stereo_geod_th).astype(float)

        else:
            raise NotImplementedError

        return xs, R, t, outliers_mask, snn_ratio, e_gt

    @classmethod
    @abstractmethod
    def get_data_file_path(cls, data_path, data_type, desc_name):
        pass

    @classmethod
    @abstractmethod
    def get_data_len(cls, data_path, data_type, desc_name):
        pass


class SUN3DDataset(RealStereoDataset):
    @classmethod
    def get_data_file_path(cls, data_path, data_type, desc_name):
        return os.path.join(data_path, f"sun3d-{desc_name}-{data_type}.hdf5")

    @classmethod
    def get_data_len(cls, data_path, data_type, desc_name):
        # Calculating the length of the training sets take ~ 6 min so store it and check it didn't change (len(self.data['xs']))
        all_data_length = dict(train=960568, test=14872, val=21484, testknown=21458)
        data_len = all_data_length[data_type]
        with h5py.File(cls.get_data_file_path(data_path, data_type, desc_name), 'r') as data_file:
            assert str(data_len - 1) in data_file['xs'].keys()
            assert str(data_len) not in data_file['xs'].keys()
        return data_len


class YFCCDataset(RealStereoDataset):
    @classmethod
    def get_data_file_path(cls, data_path, data_type, desc_name):
        return os.path.join(data_path, f"yfcc-{desc_name}-{data_type}.hdf5")

    @classmethod
    def get_data_len(cls, data_path, data_type, desc_name):
        # Calculating the length of the training sets take ~ 6 min so store it and check it didn't change (len(self.data['xs']))
        all_data_length = dict(train=541172, test=4000, val=6694, testknown=6700)
        data_len = all_data_length[data_type]
        with h5py.File(cls.get_data_file_path(data_path, data_type, desc_name), 'r') as data_file:
            assert str(data_len - 1) in data_file['xs'].keys()
            assert str(data_len) not in data_file['xs'].keys()
        return data_len

    @classmethod
    def get_noise_mean_n_std(cls, args):
        assert args.stereo_geod_method == "RepErr" and args.stereo_geod_th == 3e-3, "The noise mean and std taken under this configuration"
        noise_mean = torch.tensor([-7.7393e-06, -8.3193e-06, -7.9905e-06, -1.1421e-05])
        noise_std = torch.tensor([0.0007, 0.0010, 0.0007, 0.0010])
        return noise_mean, noise_std


class NoiseNormalization:
    def __init__(self, mean, std):
        self.mean = mean.unsqueeze(0)
        self.std = std.unsqueeze(0)

    def __call__(self, x):
        assert len(x.shape) == 2
        assert x.shape[-1] == 4
        return x * self.std.to(x.device) + self.mean.to(x.device)
