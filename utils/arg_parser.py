import configargparse
from datetime import datetime
from utils.print_utils import argparse_bool


def get_config_parser():
    parser = configargparse.ArgParser()

    # Experiment setting
    parser.add_argument('--conf', is_config_file=True, help='config file path')
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--resume', type=argparse_bool, default=False)

    # Data args
    parser.add_argument('--data_type', type=str, default="YFCC")
    parser.add_argument('--desc_name', type=str, default="sift-2000")
    parser.add_argument('--noise_free', type=argparse_bool, default=False)

    # Real world data args
    parser.add_argument("--stereo_geod_th", type=float, default=3e-3, help="theshold for the good geodesic distance")
    parser.add_argument("--stereo_geod_method", type=str, default="RepErr")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--snn_threshold', type=float, default=1)

    # Train args
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scheduler_patience', type=int, default=10)
    parser.add_argument('--scheduler_type', type=str, default="MultiStepLR")
    parser.add_argument('--scheduler_milestone', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_loader_workers', type=int, default=8)
    parser.add_argument('--freeze_encoder', type=argparse_bool, default=False)
    parser.add_argument('--detach_crct_pts', type=argparse_bool, default=False)

    # Model args
    parser.add_argument('--encoder_type', type=str, default='SetNet')
    parser.add_argument('--encoder_dim', type=int, default=512)
    parser.add_argument('--n_blocks', type=int, default=6)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--strict_model_loading', type=argparse_bool, default=True)
    parser.add_argument('--eval_from_samples', type=argparse_bool, default=False)
    parser.add_argument('--pred_from_samples', type=argparse_bool, default=True)
    parser.add_argument('--n_eor_blocks', type=int, default=3)
    parser.add_argument('--n_mlp_blocks', type=int, default=2)
    parser.add_argument('--noise_norm_func', type=str, default='max_noise')
    parser.add_argument('--pruning_ths', type=float, default=1)

    # SetNet args
    parser.add_argument('--setnet_beta', type=float, default=1)

    # Loss args
    parser.add_argument('--reg_loss_weight', type=float, default=1)
    parser.add_argument('--noise_loss_weight', type=float, default=0)
    parser.add_argument('--class_loss_weight', type=float, default=10)
    parser.add_argument('--inliers_cls_loss_weight', type=float, default=1)
    parser.add_argument('--loss_type', type=str, default="SED")
    parser.add_argument('--gradient_reduction', type=str, default="clip")
    parser.add_argument("--reg_loss_init_iter", type=int, default=0, help="initial iterations to run only the classification loss")
    parser.add_argument("--geo_loss_margin", type=float, default=0.1, help="clamping margin in geometry loss")
    parser.add_argument("--sqrt_noise_loss", type=argparse_bool, default=True)
    parser.add_argument('--noise_calc_func', type=str, default="euc_dist")
    parser.add_argument("--n_blocks_in_loss", type=int, default=-1)

    # Logger args
    parser.add_argument('--offline_logger', type=argparse_bool, default=False)

    # Eval
    parser.add_argument('--ransac_in_eval', type=argparse_bool, default=False)
    parser.add_argument('--eval_knwon_scenes', type=argparse_bool, default=False)

    return parser


def create_exp_version():
    timestamp = datetime.now()
    return timestamp.strftime("%d_%m_%Y_%H_%M_%S")


def get_args(args_str=None):
    parser = get_config_parser()
    args = parser.parse_args(args_str)
    assert args.reg_loss_weight == 1 if args.gradient_reduction == 'norm' else True, "Reg loss weight is meaningless while using norm gradient reduction"

    # Set experiments version
    if args.version is None:
        args.version = create_exp_version()

    return args



