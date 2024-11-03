import os.path
from models.Models import RobustModel
from Datasets import dataset_utils
from utils import arg_parser, trainer_utils, path_utils, print_utils
import lightning.pytorch as pl


def train_model(args, logger, ckpt_path=None):
    # Get model
    if args.pretrained_model is not None and ckpt_path is None:
        pretrained_model_path = os.path.join(path_utils.get_results_path(), args.pretrained_model)
        model = RobustModel.load_from_checkpoint(pretrained_model_path, args=args, strict=args.strict_model_loading)
        print("Loaded pretrained model from: ", pretrained_model_path)
    else:
        model = RobustModel(args)

    # Get data
    train_loader, val_loader, test_loader = dataset_utils.get_dataloaders(args)

    # Get trainer
    trainer = trainer_utils.get_trainer(args, logger)

    # Update args
    print_utils.print_exp_details(args)

    # Train
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


def main():
    args = arg_parser.get_args()

    # Resume run
    ckpt_path = None
    if args.resume:
        ckpt_path = path_utils.get_last_model_path(args)
        if ckpt_path is not None:
            args = print_utils.load_exp_details(args.run_name, args.version)

    logger = pl.loggers.CSVLogger(path_utils.get_exp_path(args))
    logger.log_hyperparams(args)

    train_model(args, logger, ckpt_path)


if __name__ == "__main__":
    main()



