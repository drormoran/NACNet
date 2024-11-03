from Datasets.BaseDataset import init_dataloader
from Datasets.StereoDataset import YFCCDataset, SUN3DDataset


def get_dataset_class(args):
    if args.data_type == "YFCC":
        return YFCCDataset
    elif args.data_type == "SUN3D":
        return SUN3DDataset
    else:
        raise NotImplementedError


def get_dataloaders(args, val_batch_size=32):
    dataset_class = get_dataset_class(args)
    train_ds = dataset_class('train', args)
    val_ds = dataset_class('val', args)
    test_ds = dataset_class('test', args)

    train_loader = init_dataloader(train_ds, args.batch_size, args.n_loader_workers)
    val_loader = init_dataloader(val_ds, val_batch_size, args.n_loader_workers)
    test_loader = init_dataloader(test_ds, val_batch_size, args.n_loader_workers)
    return train_loader, val_loader, test_loader


def get_test_dataloader(args, known_scenes=False, batch_size=32):
    dataset_class = get_dataset_class(args)
    if known_scenes:
        test_ds = dataset_class('testknown', args)
    else:
        test_ds = dataset_class('test', args)

    test_loader = init_dataloader(test_ds, batch_size, args.n_loader_workers)
    return test_loader


def get_input_dim(args):
    return get_dataset_class(args).get_input_dim()


def get_output_dim(args):
    return get_dataset_class(args).get_output_dim()


def get_normalization_params(args):
    return get_dataset_class(args).get_normalization_params(args)


def get_outliers_rate(args):
    return get_dataset_class(args).get_outliers_rate(args)


def get_meta_data(args, batch):
    return get_dataset_class(args).get_meta_data(args, batch)


def get_noise_dist(args, batch):
    return get_dataset_class(args).get_noise_dist(args, batch)


def get_strict_estimator(args):
    return get_dataset_class(args).get_strict_estimator(args)


def prepare_model_input(args, x, supp_data):
    return get_dataset_class(args).prepare_model_input(args, x, supp_data)


def restore_model_output(args, pred_shape, pred_outliers, resotre_args):
    return get_dataset_class(args).restore_model_output(args, pred_shape, pred_outliers, resotre_args)


def get_eval_regression_func(args):
    return get_dataset_class(args).get_eval_regression_func(args)


def get_eval_noise_func(args):
    return get_dataset_class(args).get_eval_noise_func(args)


def get_plot_func(args):
    return get_dataset_class(args).get_plot_func(args)


def get_global_eval_func(args):
    return get_dataset_class(args).get_global_eval_func(args)


def get_regression_loss(args):
    return get_dataset_class(args).get_regression_loss(args)


def get_noise_loss(args):
    return get_dataset_class(args).get_noise_loss(args)


def get_features_head_dims(args):
    return get_dataset_class(args).get_features_head_dims(args)


def get_noise_norm_func(args):
    return get_dataset_class(args).get_noise_norm_func(args)


def get_plot_order_dict(args, err_dict):
    return get_dataset_class(args).get_plot_order_dict(args, err_dict)
