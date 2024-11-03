from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from utils import TensorSet
from models import Losses


np.random.seed(0)


class BaseDataset(Dataset, ABC):
    def __init__(self, data_type, args):
        super(BaseDataset, self).__init__()
        self.data_type = data_type

    @abstractmethod
    def __len__(self):
        pass

    def __getitem__(self, idx):
        shape, sampled_pts, outliers_mask, supp_data = self.get_item(idx)
        return shape, sampled_pts, outliers_mask, supp_data

    @abstractmethod
    def get_item(self, idx):
        pass

    @ classmethod
    def collate_sets(cls, batch):
        return TensorSet.collate_sets(batch)

    @staticmethod
    @abstractmethod
    def get_input_dim():
        pass

    @staticmethod
    @abstractmethod
    def get_output_dim():
        pass

    @classmethod
    @abstractmethod
    def get_outliers_rate(cls, args):
        pass

    @classmethod
    @abstractmethod
    def get_eval_regression_func(cls, args):
        pass

    @classmethod
    @abstractmethod
    def get_eval_noise_func(cls, args):
        pass

    @classmethod
    @abstractmethod
    def get_plot_func(cls, args):
        pass

    @classmethod
    def get_normalization_params(cls, args):
        return cls.dummy_normalization()

    @classmethod
    def dummy_normalization(cls):
        mean = torch.zeros(cls.get_output_dim())
        std = torch.ones(cls.get_output_dim())
        return mean, std

    @classmethod
    def per_param_normalization(cls, args):
        assert str(cls.prepare_model_input).split("<class")[0] == str(BaseDataset.prepare_model_input).split("<class")[0],\
            "This solution is doesn't support model input preparation"
        data_loader = init_dataloader(cls('train', args), 1000, args.n_loader_workers)
        all_shapes = next(iter(data_loader))[0]
        mean = all_shapes.mean(dim=0)
        std = all_shapes.std(dim=0)
        return mean, std

    @classmethod
    def get_meta_data(cls, args, batch):
        meta_dict = {}
        gt_shape, sampled_pts, gt_outliers, _ = batch
        gt_inliers = gt_outliers.apply_func(lambda x: 1 - x)
        meta_dict['meta_n_inliers'] = gt_inliers.sum(dim=1).squeeze(-1).cpu()
        meta_dict['meta_outliers_rate'] = gt_outliers.mean(dim=1).squeeze(-1).cpu()
        return meta_dict

    @classmethod
    @abstractmethod
    def get_noise_dist(cls, args, batch):
        pass

    @classmethod
    def get_strict_estimator(cls, args):
        raise NotImplementedError

    @classmethod
    def prepare_model_input(cls, args, x, supp_data):
        return x, {}

    @classmethod
    def restore_model_output(cls, args, pred_shape, pred_outliers, resotre_args):
        return pred_shape, pred_outliers, resotre_args['corrected_pts'], {}

    @classmethod
    def to_torch(cls, shape, sampled_pts, outliers_mask, supp_data):
        shape = torch.from_numpy(shape).float()
        sampled_pts = torch.from_numpy(sampled_pts).float()
        outliers_mask = torch.from_numpy(outliers_mask).float()
        torch_supp_data = {}
        for key, val in supp_data.items():
            if isinstance(supp_data[key], np.ndarray):
                torch_supp_data[key] = torch.from_numpy(val).float()
            else:
                torch_supp_data[key] = val

        return shape, sampled_pts, outliers_mask, torch_supp_data

    @classmethod
    def get_global_eval_func(cls, args):
        return lambda err_dict, calc_val_loss: {}

    @classmethod
    def get_regression_loss(cls, args):
        return Losses.MSELoss()

    @classmethod
    @abstractmethod
    def get_noise_loss(cls, args):
        pass

    @classmethod
    def get_features_head_dims(cls, args):
        return 1

    @classmethod
    @abstractmethod
    def get_noise_norm_func(cls, args):
        pass

    @classmethod
    @abstractmethod
    def get_plot_order_dict(cls, args, err_dict):
        pass


class BaseImageLoader(ABC):
    @abstractmethod
    def load(self, idx):
        pass


def init_dataloader(dataset, batch_size, num_workers):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_sets, num_workers=num_workers, shuffle=True)
