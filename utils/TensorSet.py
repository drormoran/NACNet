import torch
from torch.nn import functional as F
import kornia


def collate_sets(batch, supp_data_collate_func=None):
    all_shapes = []
    all_sampled_pts = []
    all_outliers_mask = []
    all_supp_data = {}
    supp_data_keys = None

    for shape, sampled_pts, outliers_mask, supp_data in batch:
        all_shapes.append(shape)
        all_sampled_pts.append(sampled_pts)
        all_outliers_mask.append(outliers_mask)

        # Initialize supp data keys
        if supp_data_keys is None:
            supp_data_keys = supp_data.keys()
            for key in supp_data_keys:
                all_supp_data[key] = []

        # Get supplementary data
        for key, val in supp_data.items():
            all_supp_data[key].append(val)

    all_shapes = torch.stack(all_shapes)
    all_sampled_pts = TensorSet(all_sampled_pts)
    all_outliers_mask = TensorSet(all_outliers_mask)
    for key, val in all_supp_data.items():
        if supp_data_collate_func is not None:
            all_supp_data[key] = supp_data_collate_func[key](val)
        else:
            all_supp_data[key] = TensorSet(val)

    return all_shapes, all_sampled_pts, all_outliers_mask, all_supp_data


def collate_sets_batches(batches):
    all_shapes = []
    all_sampled_pts = []
    all_outliers_mask = []
    all_supp_data = {}
    supp_data_keys = None

    for shape, sampled_pts, outliers_mask, supp_data in batches:
        all_shapes.append(shape)
        all_sampled_pts.extend(sampled_pts.to_tensors_list())
        all_outliers_mask.extend(outliers_mask.to_tensors_list())

        # Initialize supp data keys
        if supp_data_keys is None:
            supp_data_keys = supp_data.keys()
            for key in supp_data_keys:
                all_supp_data[key] = []

        # Get supplementary data
        for key, val in supp_data.items():
            all_supp_data[key].extend(val.to_tensors_list())

    all_shapes = torch.cat(all_shapes)
    all_sampled_pts = TensorSet(all_sampled_pts)
    all_outliers_mask = TensorSet(all_outliers_mask)
    for key, val in all_supp_data.items():
        all_supp_data[key] = TensorSet(val)

    return all_shapes, all_sampled_pts, all_outliers_mask, all_supp_data


class TensorSet:
    def __init__(self, tensor_lists=None, indices=None, values=None, batch_size=None):
        if tensor_lists is not None:
            self.device = tensor_lists[0].device
            self.batch_size = len(tensor_lists)

            all_indices = []
            for i, tensor_set in enumerate(tensor_lists):
                indices = torch.ones(tensor_set.shape[0], device=self.device, dtype=torch.int) * i
                all_indices.append(indices)

            self.indices = torch.cat(all_indices)
            self.values = torch.cat(tensor_lists)
        else:
            assert indices is not None and values is not None and batch_size is not None
            assert (indices == indices.sort()[0]).all()  # Check indices are ordered
            self.device = values.device
            self.values = values
            self.indices = indices
            self.batch_size = batch_size

    @property
    def sets_size(self):
        all_sets_size = torch.zeros(self.batch_size, device=self.device).long()
        unique_idx, unique_count = self.indices.unique(return_counts=True)
        all_sets_size[unique_idx] = unique_count
        return all_sets_size

    @property
    def features_size(self):
        return self.values.shape[-1]

    @property
    def dtype(self):
        return self.values.dtype

    def sum(self, dim):
        if dim == 1:
            sets_sum = torch.zeros(self.batch_size, self.features_size, device=self.device, dtype=self.values.dtype)
            return sets_sum.index_add(0, self.indices, self.values)
        elif dim == 2:
            new_val = self.values.sum(dim=1, keepdim=True)
            return TensorSet(indices=self.indices.clone(), values=new_val, batch_size=self.batch_size)
        else:
            raise NotImplementedError

    def mean(self, dim):
        assert dim == 1
        return self.sum(dim=1) / (self.sets_size.unsqueeze(-1) + 1e-10)

    def std(self, dim, correction=1):
        assert dim == 1
        var = ((self - self.mean(dim=1).unsqueeze(1)) ** 2).sum(dim=1) / (self.sets_size.unsqueeze(1) - correction)
        return var.sqrt()

    def masked_mean(self, mask, eps=1e-8):
        masked_sum = (self * mask).sum(dim=1)
        masked_mean = masked_sum / (mask.sum(dim=1) + eps)
        return masked_mean.squeeze(-1)

    def to(self, device, **kwargs):
        self.device = device
        self.values = self.values.to(device, **kwargs)
        self.indices = self.indices.to(device, **kwargs)
        return self

    def detach(self):
        return TensorSet(indices=self.indices.detach(), values=self.values.detach(), batch_size=self.batch_size)

    def astype(self, new_type):
        new_val = self.values.to(new_type)
        return TensorSet(indices=self.indices.clone(), values=new_val, batch_size=self.batch_size)

    def sqrt(self):
        return self.apply_func(torch.sqrt)

    def __pow__(self, power, modulo=None):
        if modulo is not None:
            raise NotImplementedError

        new_val = self.values.__pow__(power)
        return TensorSet(indices=self.indices.clone(), values=new_val, batch_size=self.batch_size)

    def __add__(self, other):
        return self.apply_arithmetic_operation(other, '__add__')

    def __sub__(self, other):
        return self.apply_arithmetic_operation(other, '__sub__')

    def __truediv__(self, other):
        return self.apply_arithmetic_operation(other, '__truediv__')

    def __mul__(self, other):
        return self.apply_arithmetic_operation(other, '__mul__')

    def __lt__(self, other):
        return self.apply_arithmetic_operation(other, '__lt__')

    def __gt__(self, other):
        return self.apply_arithmetic_operation(other, '__gt__')

    def __le__(self, other):
        return self.apply_arithmetic_operation(other, '__le__')

    def expand(self, size):
        new_val = self.values.expand(-1, size)
        return TensorSet(indices=self.indices.clone(), values=new_val, batch_size=self.batch_size)

    def apply_arithmetic_operation(self, other, func):
        if isinstance(other, TensorSet):
            assert (self.indices == other.indices).all()
            new_values = self._apply_on_values(other.values, func)
            return TensorSet(indices=self.indices.clone(), values=new_values, batch_size=self.batch_size)

        elif isinstance(other, torch.Tensor):
            if other.shape[0] == 1:
                other = other.expand(self.batch_size, -1, -1)
            assert tuple(other.shape) == (self.batch_size, 1, self.features_size)
            new_values = self._apply_on_values(other.squeeze(1).repeat_interleave(self.sets_size, dim=0), func)
            return TensorSet(indices=self.indices.clone(), values=new_values, batch_size=self.batch_size)

        elif isinstance(other, int) or isinstance(other, float):
            new_values = self._apply_on_values(other, func)
            return TensorSet(indices=self.indices.clone(), values=new_values, batch_size=self.batch_size)

        else:
            raise NotImplementedError

    def _apply_on_values(self, other, func):
        if isinstance(other, torch.Tensor):
            assert other.shape == self.values.shape
        else:
            assert isinstance(other, int) or isinstance(other, float)
        return getattr(self.values, func)(other)

    def apply_func(self, func):
        new_values = func(self.values)
        return TensorSet(indices=self.indices.clone(), values=new_values, batch_size=self.batch_size)

    def apply_layer(self, layer_func, on_global=False):
        if on_global:
            global_feats = self.get_global_feats(True)
            return layer_func(global_feats)
        else:
            new_values = layer_func(self.values)
            return TensorSet(indices=self.indices.clone(), values=new_values, batch_size=self.batch_size)

    def clone(self):
        return TensorSet(indices=self.indices.clone(), values=self.values.clone(), batch_size=self.batch_size)

    def get_global_feats(self, keepdim=False):
        global_feats = self.mean(dim=1)
        if keepdim:
            global_feats = global_feats.unsqueeze(1)

        return global_feats

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if 0 <= idx < self.batch_size:
                return self.values[self.indices == idx]
            else:
                raise IndexError
        elif isinstance(idx, TensorSet):
            assert (self.indices == idx.indices).all()
            assert torch.logical_or(idx.values == 0, idx.values == 1).all()
            good_indices = idx.values.bool().squeeze(-1)
            assert len(good_indices.shape) == 1
            new_val = self.values[good_indices]
            new_indices = self.indices[good_indices]
            return TensorSet(indices=new_indices.clone(), values=new_val.clone(), batch_size=self.batch_size)
        elif isinstance(idx, tuple) and len(idx) == 3 and idx[0] == slice(None) and idx[1] == slice(None):
            new_val = self.values[:, idx[2]].clone()
            if len(new_val.shape) == 1:
                new_val = new_val.unsqueeze(1)
            return TensorSet(indices=self.indices.clone(), values=new_val, batch_size=self.batch_size)
        elif isinstance(idx, torch.Tensor):
            assert len(idx.shape) == 1
            assert idx.shape[0] == self.batch_size
            assert (self.indices == self.indices.sort()[0]).all()  # Check indices are ordered
            assert idx.dtype == torch.bool

            new_val = self.values[idx.repeat_interleave(self.sets_size)]
            new_sets_size = self.sets_size[idx]
            new_batch_size = idx.sum()
            new_indices = torch.arange(new_batch_size).to(new_sets_size.device).repeat_interleave(new_sets_size)
            return TensorSet(indices=new_indices, values=new_val, batch_size=new_batch_size)
        else:
            raise NotImplementedError

    def __setitem__(self, idx, new_val):
        if isinstance(idx, int) and isinstance(new_val, torch.Tensor):
            if 0 <= idx < self.batch_size:
                self.values[self.indices == idx] = new_val
            else:
                raise IndexError

        elif isinstance(idx, TensorSet) and isinstance(new_val, TensorSet):
            assert (self.indices == idx.indices).all()
            assert torch.logical_or(idx.values == 0, idx.values == 1).all()
            assert (idx[idx].indices == new_val.indices).all()
            good_indices = idx.values.bool().squeeze(-1)
            assert len(good_indices.shape) == 1
            self.values[good_indices] = new_val.values

        elif isinstance(idx, TensorSet) and isinstance(new_val, int):
            assert (self.indices == idx.indices).all()
            assert torch.logical_or(idx.values == 0, idx.values == 1).all()
            good_indices = idx.values.bool().squeeze(-1)
            assert len(good_indices.shape) == 1
            self.values[good_indices] = new_val

        else:
            raise NotImplementedError

    def to_tensors_list(self):
        assert (self.indices == self.indices.sort()[0]).all()  # Check indices are ordered
        split_size_or_sections = self.indices.unique(sorted=True, return_counts=True)[1].tolist()
        return self.values.clone().split(split_size_or_sections, dim=0)

    def all(self):
        return self.values.all()

    def clamp(self, min, max):
        new_val = self.values.clamp(min, max)
        return TensorSet(indices=self.indices.clone(), values=new_val, batch_size=self.batch_size)


def ones_like(tensor_set):
    new_val = torch.ones_like(tensor_set.values)
    return TensorSet(indices=tensor_set.indices.clone(), values=new_val, batch_size=tensor_set.batch_size)


def zeros_like(tensor_set):
    new_val = torch.zeros_like(tensor_set.values)
    return TensorSet(indices=tensor_set.indices.clone(), values=new_val, batch_size=tensor_set.batch_size)


def matches_to_points(matches):
    val1, val2 = matches.values.split(2, dim=-1)
    pts1 = TensorSet(indices=matches.indices.clone(), values=val1, batch_size=matches.batch_size)
    pts2 = TensorSet(indices=matches.indices.clone(), values=val2, batch_size=matches.batch_size)
    return pts1, pts2


def points_to_matches(pts1, pts2):
    assert (pts1.indices == pts2.indices).all()
    matches_val = torch.cat([pts1.values, pts2.values], dim=-1)
    return TensorSet(indices=pts1.indices.clone(), values=matches_val, batch_size=pts1.batch_size)


def transform_points(trans, pts):
    assert trans.shape[0] == pts.batch_size
    trans = trans.repeat_interleave(pts.sets_size, dim=0)
    new_pts_val = kornia.geometry.transform_points(trans, pts.values.unsqueeze(1)).squeeze(1)
    trans_pts = TensorSet(indices=pts.indices.clone(), values=new_pts_val, batch_size=pts.batch_size)
    return trans_pts


def oneway_transfer_error(pts1, pts2, Hs, squared, gradient_reduction, eps=1e-8):
    # Get pts1 in 2
    Hs = Hs.repeat_interleave(pts1.sets_size, dim=0)
    pts1_h = kornia.geometry.convert_points_to_homogeneous(pts1.values)
    pts1_in_2_h = torch.bmm(Hs, pts1_h.unsqueeze(-1)).squeeze(-1)

    # Convert from homogeneous with gradient normalization
    if gradient_reduction == 'norm':
        gradient_norm_denominator = pts1.sets_size.repeat_interleave(pts1.sets_size, dim=0) * pts1.batch_size
        gradient_norm_denominator = gradient_norm_denominator.unsqueeze(-1)
        pts1_in_2_h.register_hook(lambda grad: F.normalize(grad, dim=-1) / gradient_norm_denominator)

    pts1_in_2 = kornia.geometry.convert_points_from_homogeneous(pts1_in_2_h[None]).squeeze(0)

    # Calc Error
    err = (pts1_in_2 - pts2.values).pow(2).sum(dim=-1)
    if not squared:
        err = (err + eps).sqrt()

    return TensorSet(indices=pts1.indices.clone(), values=err.unsqueeze(-1), batch_size=pts1.batch_size)


def calc_stereo_error(pts1, pts2, Es, geo_func, **geo_func_kwargs):
    t_pts1 = pts1.values.unsqueeze(1)
    t_pts2 = pts2.values.unsqueeze(1)
    t_Es = Es.repeat_interleave(pts1.sets_size, dim=0)
    stereo_err = geo_func(t_pts1, t_pts2, t_Es, **geo_func_kwargs)
    return TensorSet(indices=pts1.indices.clone(), values=stereo_err, batch_size=pts1.batch_size)


def binary_cross_entropy(intput, target):
    assert (intput.values >= 0).all() and (intput.values <= 1).all(), intput.values[torch.logical_or(intput.values <= 0, intput.values >= 1)]
    assert (target.values >= 0).all() and (target.values <= 1).all(), target.values[torch.logical_or(target.values <= 0, target.values >= 1)]
    err = torch.nn.functional.binary_cross_entropy(intput.values, target.values, reduction='none')
    return TensorSet(indices=intput.indices.clone(), values=err, batch_size=intput.batch_size)


def cat(x_list, dim):
    assert dim == -1 or dim == 2
    base_x = x_list.pop(0)
    x_vals = [base_x.values]
    for x in x_list:
        assert (base_x.indices == x.indices).all()
        assert base_x.batch_size == x.batch_size
        x_vals.append(x.values)

    new_val = torch.cat(x_vals, dim=1)
    return TensorSet(indices=base_x.indices.clone(), values=new_val, batch_size=base_x.batch_size)


def project_3d_pts(pts3d, Pmat, gradient_reduction):
    Pmat = Pmat.repeat_interleave(pts3d.sets_size, dim=0)
    pts3d_h = kornia.geometry.convert_points_to_homogeneous(pts3d.values)
    pts2d_h = torch.bmm(Pmat, pts3d_h.unsqueeze(-1)).squeeze(-1)

    # Gradient normalization
    if gradient_reduction == 'norm':
        gradient_norm_denominator = pts3d.sets_size.repeat_interleave(pts3d.sets_size, dim=0) * pts3d.batch_size
        gradient_norm_denominator = gradient_norm_denominator.unsqueeze(-1)
        pts2d_h.register_hook(lambda grad: F.normalize(grad, dim=-1) / gradient_norm_denominator)

    depth = pts2d_h[:, 2:]
    pts2d = kornia.geometry.convert_points_from_homogeneous(pts2d_h.unsqueeze(0)).squeeze(0)

    depth = TensorSet(indices=pts3d.indices.clone(), values=depth, batch_size=pts3d.batch_size)
    pts2d = TensorSet(indices=pts3d.indices.clone(), values=pts2d, batch_size=pts3d.batch_size)

    return pts2d, depth


def where(condition, input, other):
    new_val = torch.where(condition.values, input.values, other.values)
    return TensorSet(indices=condition.indices.clone(), values=new_val, batch_size=condition.batch_size)


def test_set():
    s1 = torch.randn((500, 2)).double()
    s2 = torch.randn((500, 2)).double()
    s3 = torch.randn((500, 2)).double()
    s_mat = torch.stack((s1, s2, s3))
    tensor_set = TensorSet([s1, s2, s3])

    # Sum and mean
    assert torch.allclose(s_mat.sum(dim=1), tensor_set.sum(dim=1), atol=1e-7)
    assert torch.allclose(s_mat.mean(dim=1), tensor_set.mean(dim=1), atol=1e-7)
    assert torch.allclose(s_mat.std(dim=1), tensor_set.std(dim=1), atol=1e-7)

    # Global features
    global_feats = s_mat.mean(dim=1, keepdim=True)
    assert torch.allclose(global_feats, tensor_set.get_global_feats(True))

    # Add
    assert torch.allclose((s_mat + s_mat).mean(dim=1), (tensor_set + tensor_set).mean(dim=1), atol=1e-7)
    assert torch.allclose((s_mat + global_feats).mean(dim=1), (tensor_set + global_feats).mean(dim=1), atol=1e-7)

    # Subtract
    assert torch.allclose((s_mat - s_mat).mean(dim=1), (tensor_set - tensor_set).mean(dim=1), atol=1e-7)
    assert torch.allclose((s_mat - global_feats).mean(dim=1), (tensor_set - global_feats).mean(dim=1), atol=1e-7)

    # Divide
    assert torch.allclose((s_mat / s_mat).mean(dim=1), (tensor_set / tensor_set).mean(dim=1), atol=1e-7)
    assert torch.allclose((s_mat / global_feats).mean(dim=1), (tensor_set / global_feats).mean(dim=1), atol=1e-7)

    # Test Get item
    idx = torch.Tensor([True, False, True]).bool()
    for i, single_set in enumerate(tensor_set[idx].to_tensors_list()):
        assert (single_set == s_mat[idx][i]).all()

    # Prepare zero size sets
    mask_values = torch.ones((tensor_set.values.shape[0], 1), device=tensor_set.device)
    mask_values[tensor_set.indices == 1] = 0
    mask = TensorSet(indices=tensor_set.indices, values=mask_values, batch_size=tensor_set.batch_size)
    masked_tensor_set = tensor_set[mask]

    # Test
    assert masked_tensor_set.sets_size[1] == 0  # sets_size
    assert (masked_tensor_set.sum(dim=1)[1] == 0).all()  # Sum
    assert masked_tensor_set[1].shape[0] == 0  # Indexing

    # Test masked mean
    mask2 = TensorSet(indices=tensor_set.indices, values=mask_values.expand(-1, 2), batch_size=tensor_set.batch_size)
    assert (tensor_set.masked_mean(mask2)[1] == 0).all()


if __name__ == "__main__":
    test_set()
