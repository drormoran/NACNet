from utils import path_utils, print_utils
import lightning.pytorch as pl
import time


def get_checkpoint_conf(args):
    return dict(
        dirpath=path_utils.get_checkpts_path(args),
        save_top_k=1,
        monitor="ValLoss",
        save_on_train_epoch_end=False,
    )


def get_last_checkpoint_conf(args):
    return dict(
        dirpath=path_utils.get_last_checkpts_path(args),
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True
    )


class PrintingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        err_dict = {k: trainer.callback_metrics[k] for k in trainer.callback_metrics.keys() if k.startswith("Val")}
        print_utils.print_erros(pl_module.args, err_dict, trainer.current_epoch)


def get_callback(args):
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(**get_checkpoint_conf(args))
    last_checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(**get_last_checkpoint_conf(args))
    exception_callback = pl.callbacks.OnExceptionCheckpoint(path_utils.get_checkpts_path(args))
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks_lists = [checkpoint_callback, lr_callback, PrintingCallback(), last_checkpoint_callback, exception_callback]

    return callbacks_lists


def get_trainer(args, logger=None):
    gradient_clip_val = 0.5 if args.gradient_reduction == 'clip' else None
    return pl.Trainer(logger=logger, callbacks=get_callback(args), max_epochs=args.max_epochs,
                      gradient_clip_val=gradient_clip_val, val_check_interval=0.5)


