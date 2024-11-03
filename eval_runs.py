from models import Models
from utils import path_utils, trainer_utils, arg_parser
from Datasets.dataset_utils import get_test_dataloader
import pandas as pd


def eval_exp(args, model):
    # Run test
    test_loader = get_test_dataloader(args, known_scenes=args.eval_knwon_scenes)
    trainer = trainer_utils.get_trainer(args)
    exp_errors = trainer.test(model, dataloaders=test_loader)

    # Combine all errors
    samples_err_pd = pd.DataFrame.from_dict(trainer.model.test_step_errors)

    return exp_errors, samples_err_pd


def eval_runs():
    args = arg_parser.get_args()

    # Load model
    ckpt_path = path_utils.get_best_model_path(args)
    print(f"Loading model from: {ckpt_path}")
    model = Models.RobustModel.load_from_checkpoint(ckpt_path, args=args)

    print(f"# # # # # # # # Evaluating {args.run_name}:{args.version} # # # # # # # # ")
    exp_errors, samples_errors = eval_exp(args, model)

    # Get exp errors
    exp_errors_pd = pd.DataFrame.from_dict(exp_errors)
    exp_errors_pd.index = [args.run_name]


if __name__ == "__main__":
    eval_runs()