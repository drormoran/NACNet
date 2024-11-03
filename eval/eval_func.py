import torch
from datetime import datetime
from utils import path_utils, print_utils, arg_parser
import os


def eval_classification(b_pred_outliers, gt_outliers):
    b_pred_inliers = b_pred_outliers.apply_func(lambda x: 1 - x)
    gt_inliers = gt_outliers.apply_func(lambda x: 1 - x)

    # Calc precision and IoU
    TP = b_pred_inliers[gt_inliers].sum(dim=1).squeeze(1)
    FP = b_pred_inliers[gt_outliers].sum(dim=1).squeeze(1)
    FN = b_pred_outliers[gt_inliers].sum(dim=1).squeeze(1)

    iou_val = TP / (TP + FP + FN + 1e-10)
    precision_val = TP / (TP + FP + 1e-10)
    recall_val = TP / (TP + FN + 1e-10)
    f1_val = 2 * TP / (2 * TP + FN + FP + 1e-10)

    # Set values to zero for samples with less than 4 inliers
    n_pred_inliers = b_pred_inliers.sum(dim=1).squeeze(1)
    iou_val[n_pred_inliers < 4] = 0
    precision_val[n_pred_inliers < 4] = 0
    recall_val[n_pred_inliers < 4] = 0
    f1_val[n_pred_inliers < 4] = 0

    return dict(IoU=iou_val.cpu(), Precision=precision_val.cpu(), Recall=recall_val.cpu(), F1=f1_val.cpu())


def get_eval_class_keys():
    return ["IoU", "Precision", "Recall", "F1"]


def rad2deg(theta):
    theta = theta % (2 * torch.pi)
    return torch.rad2deg(theta)


def concat_dict(dic1, dic2):
    new_dic = dic1.copy()
    for key, val in dic2.items():
        if key in dic1.keys():
            new_dic[key] = torch.cat((dic1[key], dic2[key]), dim=0)
        else:
            new_dic[key] = dic2[key]

    return new_dic


def get_latest_version(exp_name):
    exp_parent_path = path_utils.get_exp_parent_path(exp_name)
    latest_timestamp, version = None, None
    for exp_version in os.listdir(exp_parent_path):
        exp_timestamp = datetime.strptime(exp_version, "%d_%m_%Y_%H_%M_%S")
        if latest_timestamp is None or exp_timestamp > latest_timestamp:
            version = exp_version
            latest_timestamp = exp_timestamp
    print(f"Using latest version: {latest_timestamp}")

    return version, latest_timestamp




