import numpy as np
import cv2
import kornia
import torch
from utils import TensorSet


class EvalGlobalErr:
    def __init__(self, args):
        self.maa_ths = torch.arange(7).float() * 5
        self.auc_ths = [5, 10, 20]

    def get_hist(self, x):
        num_pair = x.shape[0]
        x_acc_hist, _ = torch.histogram(x, self.maa_ths)
        x_acc_hist = x_acc_hist / num_pair
        x_acc = torch.cumsum(x_acc_hist, dim=0)
        return x_acc

    def pose_auc(self, errors, ths):
        errors = errors.sort()[0]
        recall = (torch.arange(errors.shape[0]) + 1) / errors.shape[0]
        errors = torch.cat((torch.zeros((1), device=errors.device), errors))
        recall = torch.cat((torch.zeros((1), device=recall.device), recall))
        last_index = torch.searchsorted(errors, ths)

        if last_index != 0:
            r = torch.cat((recall[:last_index], recall[last_index - 1].reshape(-1)))
            e = torch.cat((errors[:last_index], torch.tensor([ths], device=errors.device)))
            auc = torch.trapz(r, x=e) / ths
            auc = auc.item()
        else:
            auc = 0

        return auc

    def __call__(self, err_dict, calc_val_loss):
        # Calculate AUC and mAA
        err_q = err_dict['err_q']
        err_t = err_dict['err_t']
        err_qt = torch.maximum(err_q, err_t)

        # Get histogram
        q_acc = self.get_hist(err_q)
        t_acc = self.get_hist(err_t)
        qt_acc = self.get_hist(err_qt)

        # Calculate mAP
        err_dict = {}
        for i in range(1, self.maa_ths.shape[0]):
            err_dict[f'q_mAP_{self.maa_ths[i]}'] = q_acc[:i].mean().item()
            err_dict[f't_mAP_{self.maa_ths[i]}'] = t_acc[:i].mean().item()
            err_dict[f'qt_mAP_{self.maa_ths[i]}'] = qt_acc[:i].mean().item()

        # Calculate AUC
        for auc_threshold in self.auc_ths:
            err_dict[f'q_auc_{auc_threshold}'] = self.pose_auc(err_q, auc_threshold)
            err_dict[f't_auc_{auc_threshold}'] = self.pose_auc(err_t, auc_threshold)
            err_dict[f'qt_auc_{auc_threshold}'] = self.pose_auc(err_qt, auc_threshold)

        if calc_val_loss:
            err_dict['ValLoss'] = -err_dict['qt_mAP_5.0']

        return err_dict


def eval_essential_mat(b_pred_shape, b_pred_outliers, batch):
    _, sampled_pts, gt_outliers, supp_data = batch
    b_pred_E = shape_to_E(b_pred_shape)

    # Get rotation and translation errors
    pred_R, pred_t = E_to_R_t(b_pred_E, sampled_pts, b_pred_outliers)
    gt_R, gt_t = supp_data['R'], supp_data['t']

    # To quaternion
    gt_q = rotation_mat_to_quaternion(gt_R)
    pred_q = rotation_mat_to_quaternion(pred_R)

    b_err_q, b_err_t = evaluate_q_t(gt_q, gt_t.squeeze(-1), pred_q, pred_t.squeeze(-1))
    b_err_q = b_err_q * 180.0 / np.pi
    b_err_t = b_err_t * 180.0 / np.pi

    err_dict = dict(err_q=b_err_q.cpu(), err_t=b_err_t.cpu())
    return err_dict


def evaluate_q_t(gt_q, gt_t, pred_q, pred_t, eps=1e-15):
    def prepare_vec(x):
        x = x / (torch.norm(x, dim=-1, keepdim=True) + eps)
        return x

    # Quaternion error
    pred_q = prepare_vec(pred_q)
    gt_q = prepare_vec(gt_q)
    loss_q = torch.maximum(torch.tensor(eps), (1.0 - torch.sum(pred_q * gt_q, dim=-1)**2))
    err_q = torch.arccos(1 - 2 * loss_q)

    # Translation error
    pred_t = prepare_vec(pred_t)
    gt_t = prepare_vec(gt_t)
    loss_t = torch.maximum(torch.tensor(eps), (1.0 - torch.sum(pred_t * gt_t, dim=-1)**2))
    err_t = torch.arccos(torch.sqrt(1 - loss_t))

    assert err_q.isnan().sum() == 0 and err_t.isnan().sum() == 0
    return err_q, err_t


class EssentialMatEstimator:
    def __init__(self, args):
        self.use_ransac = args.ransac_in_eval

    def __call__(self, sampled_pts, outliers_mask):
        pred_inliers_samples = sampled_pts[outliers_mask.apply_func(lambda x: (x == 0).float())]
        if self.use_ransac:
            return find_E_with_RANSAC(pred_inliers_samples)
        else:
            return weighted_8points(pred_inliers_samples)


def find_E_with_RANSAC(pred_inliers_samples):
    b_pred_shape = []
    for i in range(pred_inliers_samples.batch_size):
        # Get Essential matrix
        pts1, pts2 = np.split(pred_inliers_samples[i].cpu().numpy(), 2, axis=-1)
        if pts1.shape[0] >= 5:
            pred_E, mask_new = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, threshold=0.001, maxIters=1000)
            if pred_E is None:
                pred_E = calc_essential_mat(np.array([[1., 0., 0.]]).T, np.eye(3))
            elif pred_E.shape[0] > 3:
                pred_E = pred_E[:3]
            pred_E = pred_E / np.linalg.norm(pred_E, ord=2, axis=(-2, -1), keepdims=True)
            pred_shape = E_to_shape(pred_E)
        else:
            pred_E = calc_essential_mat(np.array([[1., 0., 0.]]).T, np.eye(3))
            pred_shape = E_to_shape(pred_E)

        b_pred_shape.append(torch.from_numpy(pred_shape).float())
    b_pred_shape = torch.stack(b_pred_shape).to(pred_inliers_samples.device)
    return b_pred_shape


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    device = X.device
    X = X.cpu()
    bv = torch.empty_like(X)
    for batch_idx in range(X.shape[0]):
        _, v = torch.linalg.eigh(X[batch_idx], UPLO='U')
        bv[batch_idx, :, :] = v
    bv = bv.to(device)
    return bv


def weighted_8points(x_in, out_logits=None):
    assert isinstance(x_in, TensorSet.TensorSet)

    if out_logits is None:
        out_logits = TensorSet.zeros_like(x_in[:, :, :2])

    # Get weights
    in_logits = out_logits * (-1)
    mask = in_logits[:, :, 0].apply_func(torch.sigmoid)
    weights = in_logits[:, :, 1].apply_func(torch.exp) * mask
    weights = weights / (weights.sum(dim=1).unsqueeze(1) + 1e-5)

    X = TensorSet.cat([
        x_in[:, :, 2] * x_in[:, :, 0], x_in[:, :, 2] * x_in[:, :, 1], x_in[:, :, 2],
        x_in[:, :, 3] * x_in[:, :, 0], x_in[:, :, 3] * x_in[:, :, 1], x_in[:, :, 3],
        x_in[:, :, 0], x_in[:, :, 1], TensorSet.ones_like(x_in[:, :, 0])
    ], dim=-1)
    wX = weights.expand(9) * X

    # Sets implementation for  XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)
    X_mul_wX = X.values.unsqueeze(-1) * wX.values.unsqueeze(1)
    sets_sum = torch.zeros(wX.batch_size, 9, 9, device=wX.device, dtype=wX.values.dtype)
    XwX = sets_sum.index_add(0, wX.indices, X_mul_wX)

    # Recover essential matrix from self-adjoing eigen
    e_hat = batch_symeig(XwX)[:, :, 0]
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


def calc_essential_mat(t, R):
    assert R.shape[-2:] == (3, 3) and t.shape[-2:] == (3, 1)
    is_numpy = isinstance(R, np.ndarray)

    if is_numpy:
        t = torch.from_numpy(t)
        R = torch.from_numpy(R)

    E = kornia.geometry.essential_from_Rt(torch.eye(3, device=R.device, dtype=R.dtype), torch.zeros_like(t), R, t)
    E = E / E.norm(p=2, dim=(-2, -1), keepdim=True)

    if is_numpy:
        E = E.numpy()

    return E


def rotation_mat_to_quaternion(R):
    is_numpy = False
    if isinstance(R, np.ndarray):
        is_numpy = True
        R = torch.from_numpy(R)

    q = kornia.geometry.rotation_matrix_to_quaternion(R)

    if is_numpy:
        q = q.numpy()

    return q


def quaternion_to_rotation_mat(q):
    is_numpy = False
    if isinstance(q, np.ndarray):
        is_numpy = True
        q = torch.from_numpy(q)

    R = kornia.geometry.quaternion_to_rotation_matrix(q)

    if is_numpy:
        R = R.numpy()

    return R


def E_to_R_t(E_mat, samples_pts, outliers_mask=None):
    device = E_mat.device

    # Remove outliers
    if outliers_mask is not None:
        inliers_mask = outliers_mask.apply_func(lambda x: (x == 0).float())
        samples_pts = samples_pts[inliers_mask]

    pts1, pts2 = TensorSet.matches_to_points(samples_pts)
    assert pts1.features_size == 2 and pts2.features_size == 2

    # Recover R, t
    all_R, all_t = [], []
    for i in range(E_mat.shape[0]):
        if pts1[i].shape[0] > 0:
            _, R, t, _ = cv2.recoverPose(E_mat[i].cpu().double().numpy(), pts1[i].cpu().double().numpy(), pts2[i].cpu().double().numpy())
        else:
            t = np.array([[1., 0., 0.]]).T
            R = np.eye(3)

        all_R.append(torch.from_numpy(R).float())
        all_t.append(torch.from_numpy(t).float())

    all_R = torch.stack(all_R).to(device)
    all_t = torch.stack(all_t).to(device)
    return all_R, all_t


def q_t_to_E(q, t):
    assert q.shape[-1] == 4
    assert t.shape[-1] == 3
    is_numpy = isinstance(q, np.ndarray)
    if is_numpy:
        t = np.expand_dims(t, axis=-1)
    else:
        t = t.unsqueeze(-1)

    R = quaternion_to_rotation_mat(q)
    e_mat = calc_essential_mat(t, R)

    return e_mat


def q_t_to_shape(q, t):
    E = q_t_to_E(q, t)
    return E_to_shape(E)


def shape_to_E(shape):
    assert shape.shape[-1] == 9

    if len(shape.shape) == 1:
        return shape.reshape(3, 3)
    elif len(shape.shape) == 2:
        return shape.reshape(-1, 3, 3)
    else:
        raise NotImplementedError


def E_to_shape(E_mat):
    assert E_mat.shape[-2:] == (3,3)

    if len(E_mat.shape) == 2:
        return E_mat.reshape(9)
    elif len(E_mat.shape) == 3:
        return E_mat.reshape(-1, 9)
    else:
        raise NotImplementedError


def correct_matches(e_gt, pts1, pts2):
    assert len(pts2.shape) == 2 and pts2.shape[-1] == 2
    assert len(pts1.shape) == 2 and pts1.shape[-1] == 2
    assert e_gt.shape == (3, 3)
    if pts1.shape[0] == 0:
        return pts1, pts2

    pts1, pts2 = pts1.reshape(1, -1, 2), pts2.reshape(1, -1, 2)
    pts1, pts2 = cv2.correctMatches(e_gt.reshape(3, 3), pts1, pts2)
    return pts1.squeeze(0), pts2.squeeze(0)


def triangulate_pts(pts1, pts2, R, t, return_depth=False):
    P1 = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    P2 = np.concatenate((R, t), axis=1)
    pts3d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T).T
    pts3d = pts3d[:, :3] / pts3d[:, 3:]

    if return_depth:
        depth1 = pts3d[:, 2]
        pts3d_in2 = (R @ pts3d.T + t).T
        depth2 = pts3d_in2[:, 2]
        return pts3d, depth1, depth2
        # Sanity check
        # np.abs(pts3d_in2[:, :2] / pts3d_in2[:, 2:] - pts2).max()
        # np.abs(pts3d[:, :2] / pts3d[:, 2:] - pts1).max()
    else:
        return pts3d


def calc_reprojection_error(xs, e_gt, R, t):
    in_pts1, in_pts2 = np.split(xs, 2, axis=-1)
    crct_pts1, crct_pts2 = correct_matches(e_gt, in_pts1, in_pts2)

    # Get reprojection error
    rep_err, rep_err1, rep_err2 = reprojection_err(in_pts1, crct_pts1, in_pts2, crct_pts2)

    # Get depth
    _, depth1, depth2 = triangulate_pts(crct_pts1, crct_pts2, R, t, return_depth=True)

    return rep_err, rep_err1, rep_err2, depth1, depth2


def reprojection_err(pts1, gt_pts1, pts2, gt_pts2):
    rep1 = np.linalg.norm(gt_pts1 - pts1, axis=-1, ord=2)
    rep2 = np.linalg.norm(gt_pts2 - pts2, axis=-1, ord=2)
    rep_err = (rep1 + rep2) / 2
    return rep_err, rep1, rep2


class StereoNoiseFunc:
    def __init__(self, noise_calc_func):
        self.noise_calc_func = noise_calc_func

    def __call__(self, corrected_pts, batch, b_sqrt=True):
        _, _, gt_outliers, supp_data = batch

        if self.noise_calc_func == "euc_dist":
            noise_err1, noise_err2 = self.calc_euc_dist(corrected_pts, supp_data['crct_pts'], gt_outliers, b_sqrt)
        elif self.noise_calc_func == "epipolar_dist":
            noise_err1, noise_err2 = self.calc_epipolar_distance(corrected_pts, supp_data['crct_pts'], gt_outliers, supp_data['e_gt'], b_sqrt)
        elif self.noise_calc_func == "Sampson":
            noise_err1, noise_err2 = self.calc_sampson_dist(corrected_pts, supp_data['crct_pts'], gt_outliers, supp_data['e_gt'], b_sqrt)
        elif self.noise_calc_func == "triangulated_euc_dist":
            # Get triangulated gt corrected pts
            assert corrected_pts.values.requires_grad == False, "This process is heavy and non deferentiable do not using during training"
            crct_pts1, crct_pts2 = TensorSet.matches_to_points(corrected_pts)
            trian_corrected_pts = []
            for i in range(corrected_pts.batch_size):
                tr_crct_pts1, tr_crct_pts2 = correct_matches(supp_data['e_gt'][i].cpu().numpy(), crct_pts1[i].cpu().numpy(), crct_pts2[i].cpu().numpy())
                trian_corrected_pts.append(torch.from_numpy(np.concatenate((tr_crct_pts1, tr_crct_pts2), axis=1)))
            trian_corrected_pts = TensorSet.TensorSet(trian_corrected_pts)
            trian_corrected_pts = trian_corrected_pts.to(corrected_pts.device)

            noise_err1, noise_err2 = self.calc_euc_dist(corrected_pts, trian_corrected_pts, gt_outliers, b_sqrt)
        else:
            raise NotImplementedError

        return dict(noise_err1=noise_err1, noise_err2=noise_err2)

    @staticmethod
    def calc_euc_dist(corrected_pts, gt_corrected_pts, gt_outliers, b_sqrt):
        pts_diff = (gt_corrected_pts - corrected_pts)[gt_outliers.apply_func(lambda x: x == 0)]
        euc_dist1, euc_dist2 = TensorSet.matches_to_points(pts_diff)
        euc_dist1 = euc_dist1.apply_func(lambda x: x ** 2).sum(dim=2)
        if b_sqrt:
            euc_dist1 = euc_dist1.apply_func(lambda x: torch.sqrt(x + 1e-16))

        euc_dist2 = euc_dist2.apply_func(lambda x: x ** 2).sum(dim=2)
        if b_sqrt:
            euc_dist2 = euc_dist2.apply_func(lambda x: torch.sqrt(x + 1e-16))

        return euc_dist1.mean(dim=1), euc_dist2.mean(dim=1)

    @staticmethod
    def calc_epipolar_distance(corrected_pts, gt_corrected_pts, gt_outliers, e_gt, b_sqrt):
        gt_inliers = gt_outliers.apply_func(lambda x: x == 0)
        crct_pts1, crct_pts2 = TensorSet.matches_to_points(corrected_pts[gt_inliers])
        gt_crct_pts1, gt_crct_pts2 = TensorSet.matches_to_points(gt_corrected_pts[gt_inliers])

        ep_dist1 = TensorSet.calc_stereo_error(crct_pts1, gt_crct_pts2, e_gt, kornia.geometry.right_to_left_epipolar_distance)
        ep_dist2 = TensorSet.calc_stereo_error(gt_crct_pts1, crct_pts2, e_gt, kornia.geometry.left_to_right_epipolar_distance)

        if not b_sqrt:
            ep_dist1 = ep_dist1 ** 2
            ep_dist2 = ep_dist2 ** 2

        return ep_dist1.mean(dim=1), ep_dist2.mean(dim=1)


    @staticmethod
    def calc_sampson_dist(corrected_pts, gt_corrected_pts, gt_outliers, e_gt, b_sqrt):
        if b_sqrt:
            return_squared = False
        else:
            return_squared = True

        gt_inliers = gt_outliers.apply_func(lambda x: x == 0)
        crct_pts1, crct_pts2 = TensorSet.matches_to_points(corrected_pts[gt_inliers])
        gt_crct_pts1, gt_crct_pts2 = TensorSet.matches_to_points(gt_corrected_pts[gt_inliers])

        ep_dist1 = TensorSet.calc_stereo_error(crct_pts1, gt_crct_pts2, e_gt, kornia.geometry.sampson_epipolar_distance, squared=return_squared)
        ep_dist2 = TensorSet.calc_stereo_error(gt_crct_pts1, crct_pts2, e_gt, kornia.geometry.sampson_epipolar_distance, squared=return_squared)

        return ep_dist1.mean(dim=1), ep_dist2.mean(dim=1)


