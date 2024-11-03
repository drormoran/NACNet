import numpy as np
from torch import optim, nn
import lightning.pytorch as pl
from models.SetNet import SetNet
from models.models_helper import ClassificationHead, RegressionHead, get_binary_classification, FeaturesHeads, NoiseHead
from models import Losses
from utils import path_utils, TensorSet, print_utils
from eval import eval_func
import torch
from Datasets import dataset_utils
import os
import sys
import traceback


def get_encoder(d_in, args):
    if args.encoder_type == 'SetNet':
        return SetNet(d_in, args.encoder_dim, args.n_blocks, args.setnet_beta)
    else:
        raise NotImplementedError


class RobustModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Init model
        self.eor_blocks = nn.ModuleList([EOR(args) for _ in range(args.n_eor_blocks)])

        # Get normalization params
        norm_mean, norm_std = dataset_utils.get_normalization_params(args)
        self.register_buffer('norm_mean', norm_mean[None], persistent=False)
        self.register_buffer('norm_std', norm_std[None], persistent=False)

        # Init loss
        self.class_loss = Losses.ClassficationLoss(args)
        self.reg_loss = dataset_utils.get_regression_loss(args)
        self.noise_loss = dataset_utils.get_noise_loss(args)

        # Init eval
        self.eval_reg_func = dataset_utils.get_eval_regression_func(args)
        self.eval_noise_func = dataset_utils.get_eval_noise_func(args)
        self.plot_func = dataset_utils.get_plot_func(args)
        self.strict_estimator = dataset_utils.get_strict_estimator(args)

        # For testing
        self.test_step_errors = {}
        self.val_step_errors = {}

    def training_step(self, batch, batch_idx):
        gt_shape, sampled_pts, gt_outliers, supp_data = batch

        try:
            # Get predictions
            all_supp_pred = self.get_batch_predictions(sampled_pts, supp_data)

            if self.args.n_blocks_in_loss != -1:
                all_supp_pred = all_supp_pred[-self.args.n_blocks_in_loss:]

            # Calc loss
            loss = 0
            for i, supp_pred in enumerate(all_supp_pred):
                # Classification loss
                class_loss_val = self.class_loss(supp_pred, batch)
                self.log(f"Class_loss_{i}", class_loss_val)
                loss += class_loss_val * self.args.class_loss_weight

                # Noise loss
                noise_loss_val = self.noise_loss(supp_pred, batch)
                self.log(f"Noise_loss_{i}", noise_loss_val)
                loss += noise_loss_val * self.args.noise_loss_weight

                # Regression loss
                if self.args.reg_loss_init_iter <= self.global_step:
                    reg_loss_val = self.reg_loss(batch, supp_pred)
                    self.log(f"Reg_loss_{i}", reg_loss_val)
                    loss += reg_loss_val * self.args.reg_loss_weight

        except Exception as e:
            torch.save(gt_shape, os.path.join(path_utils.get_checkpts_path(self.args), "gt_shape.pt"))
            torch.save(gt_outliers, os.path.join(path_utils.get_checkpts_path(self.args), "gt_outliers.pt"))
            torch.save(sampled_pts, os.path.join(path_utils.get_checkpts_path(self.args), "sampled_pts.pt"))
            traceback.print_exc()
            if "idx" in supp_data.keys():
                print(f"Error occured in IDX: {supp_data['idx'].cpu().numpy().tolist()}")
            sys.exit(f"Error during training step: {e}")

        # Log
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Reset validation step errors
        if batch_idx == 0:
            self.val_step_errors = {}

        # Eval
        pred_shape, b_pred_outliers, corrected_pts, err_dict = self.batch_eval(batch)

        # Log
        if self.trainer.state.stage != pl.trainer.states.RunningStage.SANITY_CHECKING:
            for key, val in err_dict.items():
                self.log(key, val.mean(), prog_bar=True, batch_size=val.shape[0])

        # Plot
        if batch_idx == 0 and self.current_epoch % 10 == 0:
            base_path = path_utils.get_visulization_path(self.args, self.current_epoch)
            self.plot_func(batch, pred_shape, b_pred_outliers, base_path, corrected_pts=corrected_pts)

        for key, val in err_dict.items():
            if key in self.val_step_errors.keys():
                self.val_step_errors[key] = torch.cat((self.val_step_errors[key], val))
            else:
                self.val_step_errors[key] = val

    def on_validation_epoch_end(self):
        global_eval_func = dataset_utils.get_global_eval_func(self.args)
        global_err_dict = global_eval_func(self.val_step_errors, True)

        # Log
        if self.trainer.state.stage != pl.trainer.states.RunningStage.SANITY_CHECKING:
            for key, val in global_err_dict.items():
                self.log(key, val, prog_bar=True, batch_size=self.val_step_errors['Precision'].shape[0])

        if self.trainer.state.stage != pl.trainer.states.RunningStage.SANITY_CHECKING:
            assert 'ValLoss' in self.trainer.logged_metrics.keys()

    def test_step(self, batch, batch_idx):
        # Eval
        if batch_idx == 0:
            self.test_step_errors = {}
        pred_shape, b_pred_outliers, corrected_pts, err_dict = self.batch_eval(batch)

        for key in eval_func.get_eval_class_keys():
            self.log(key, err_dict[key].mean(), prog_bar=True, batch_size=err_dict[key].shape[0])

        for key, val in err_dict.items():
            val = val.cpu()
            if len(val.shape) == 2:
                val = val.squeeze(1)
            if key in self.test_step_errors.keys():
                self.test_step_errors[key] = torch.cat((self.test_step_errors[key], val))
            else:
                self.test_step_errors[key] = val

    def on_test_epoch_end(self):
        # Test data type
        test_data_type = self.trainer.test_dataloaders.dataset.data_type

        global_eval_func = dataset_utils.get_global_eval_func(self.args)
        global_err_dict = global_eval_func(self.test_step_errors, True)

        # Log
        for key, val in global_err_dict.items():
            self.log(key, val, prog_bar=True)

        test_err_pd = print_utils.dict_to_pandas("TestErr", self.trainer.logged_metrics).T
        test_err_pd.to_excel(path_utils.get_model_test_err_path(self.args, test_data_type))

    def batch_eval(self, batch):
        # Get predictions
        gt_shape, sampled_pts, gt_outliers, supp_data = batch
        pred_shape, b_pred_outliers, corrected_pts = self.get_strict_batch_prediction(sampled_pts, supp_data)

        # Evaluation
        err_dict = eval_func.eval_classification(b_pred_outliers, gt_outliers)
        err_dict.update(self.eval_reg_func(pred_shape, b_pred_outliers, batch))
        return pred_shape, b_pred_outliers, corrected_pts, err_dict

    def get_strict_batch_prediction(self, sampled_pts, supp_data):
        all_supp_pred = self.get_batch_predictions(sampled_pts, supp_data)
        last_prediction = all_supp_pred[-1]
        pred_shape = last_prediction["pred_shape"]
        pred_outliers = last_prediction["pred_outliers"]
        corrected_pts = last_prediction['corrected_pts']

        b_pred_outliers = get_binary_classification(pred_outliers)

        # Get predicted shape
        if self.args.eval_from_samples:
            pred_shape = self.strict_estimator(corrected_pts, b_pred_outliers)

        return pred_shape, b_pred_outliers, corrected_pts

    def get_batch_predictions(self, sampled_pts, supp_data):
        # Prepare input accroding to the data
        sampled_pts, resotre_args = dataset_utils.prepare_model_input(self.args, sampled_pts, supp_data)
        pts_encoding = None
        all_supp_pred = []

        # Get prediction
        for i, eor_block in enumerate(self.eor_blocks):
            if self.args.pruning_ths < 1 and len(all_supp_pred) > 0:
                batch_prediction = self.get_pruned_predictions(eor_block, sampled_pts, pts_encoding, all_supp_pred[-1], i)
            else:
                batch_prediction = eor_block(sampled_pts, pts_encoding)

            pred_shape, pred_outliers, resotre_args['pred_logits'], pred_noise, pred_features, pts_encoding = batch_prediction
            resotre_args['corrected_pts'] = sampled_pts - pred_noise
            sampled_pts = resotre_args['corrected_pts'].detach() if self.args.detach_crct_pts else resotre_args['corrected_pts']

            # Normalize shape
            pred_shape = pred_shape * self.norm_std + self.norm_mean

            # Restore to original input system
            pred_shape, pred_outliers, corrected_pts, supp_pred = dataset_utils.restore_model_output(self.args, pred_shape, pred_outliers, resotre_args)
            supp_pred["pred_features"] = pred_features
            supp_pred["corrected_pts"] = corrected_pts
            supp_pred["pred_shape"] = pred_shape
            supp_pred["pred_outliers"] = pred_outliers
            all_supp_pred.append(supp_pred)

        return all_supp_pred

    def get_pruned_predictions(self, eor_block, sampled_pts, pts_encoding, prev_pred, eor_blc_idx):
        if self.args.pruning_ths < 0:
            # Pruning 0.5 of the samples
            pruning_mask = []
            for i in range(prev_pred['pred_outliers'].batch_size):
                pred_out_i = prev_pred['pred_outliers'][i]
                n_samples = int(pred_out_i.shape[0] * 0.5)
                prun_mask_i = torch.zeros_like(pred_out_i, dtype=torch.bool)
                prun_idx = pred_out_i.argsort(dim=0)[:n_samples]
                prun_mask_i[prun_idx] = True
                prun_mask_i[pred_out_i == 1] = False
                pruning_mask.append(prun_mask_i)
            pruning_mask = TensorSet.TensorSet(pruning_mask)

        else:
            pruning_mask = prev_pred['pred_outliers'].apply_func(lambda x: x <= self.args.pruning_ths)

        pruned_sampled_pts = sampled_pts[pruning_mask]
        pruned_pts_encoding = pts_encoding[pruning_mask]

        batch_prediction = eor_block(pruned_sampled_pts, pruned_pts_encoding)
        pred_shape, pruned_pred_outliers, pruned_pred_logits, pruned_pred_noise, pruned_pred_features, pruned_pts_encoding = batch_prediction

        # Restore pruned input
        pred_outliers = TensorSet.ones_like(prev_pred['pred_outliers'])
        pred_outliers[pruning_mask] = pruned_pred_outliers

        pred_logits = TensorSet.ones_like(prev_pred['pred_outliers']).expand(2) * 50
        pred_logits[pruning_mask] = pruned_pred_logits

        pred_noise = TensorSet.zeros_like(sampled_pts)
        pred_noise[pruning_mask] = pruned_pred_noise

        pred_features = TensorSet.ones_like(prev_pred['pred_features']) * torch.nan
        pred_features[pruning_mask] = pruned_pred_features

        pts_encoding = TensorSet.ones_like(pts_encoding) * torch.nan
        pts_encoding[pruning_mask] = pruned_pts_encoding

        if self.trainer.state.stage == pl.trainer.states.RunningStage.VALIDATING:
            n_pruned_pts = pruning_mask.apply_func(lambda x: (x == False).float()).sum(dim=1)
            self.log(f"N_pruned_pts{eor_blc_idx}", n_pruned_pts.mean(), prog_bar=True, batch_size=pruning_mask.batch_size)

        return pred_shape, pred_outliers, pred_logits, pred_noise, pred_features, pts_encoding

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        if self.args.scheduler_type == "MultiStepLR":
            if self.args.scheduler_milestone is not None:
                milestones = [self.args.scheduler_milestone]
            else:
                milestones = []
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, verbose=True)

        elif self.args.scheduler_type == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.args.scheduler_patience, verbose=True)

        else:
            raise NotImplementedError

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss",
            },
        }

    def on_before_optimizer_step(self, optimizer):
        norms_dict = {}
        if self.args.n_blocks_in_loss != -1:
            list_block_idx = [-1]
        else:
            list_block_idx = [0, -1]
        for i in list_block_idx:
            norms_dict[f'Cls_Head_Norm{i}'] = pl.utilities.grad_norm(self.eor_blocks[i].class_head, norm_type=2)['grad_2.0_norm_total']
            norms_dict[f'Nse_Head_Norm{i}'] = pl.utilities.grad_norm(self.eor_blocks[i].noise_head, norm_type=2)['grad_2.0_norm_total']
            if next(iter(self.eor_blocks[i].reg_head.parameters())).grad is not None:
                norms_dict[f'Reg_Head_Norm{i}'] = pl.utilities.grad_norm(self.eor_blocks[i].reg_head, norm_type=2)['grad_2.0_norm_total']

        self.log_dict(norms_dict)


class EOR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.freeze_encoder = args.freeze_encoder
        d_in = dataset_utils.get_input_dim(args)
        self.encoder = get_encoder(d_in, args)
        self.class_head = ClassificationHead(args.encoder_dim, args.n_mlp_blocks, args.setnet_beta)
        self.reg_head = RegressionHead(args.encoder_dim, dataset_utils.get_output_dim(args), args.n_mlp_blocks, args.setnet_beta)
        self.features_head = FeaturesHeads(args.encoder_dim, dataset_utils.get_features_head_dims(args), args.n_mlp_blocks, args.setnet_beta)

        if args.noise_loss_weight > 0:
            noise_norm_function = dataset_utils.get_noise_norm_func(args)
        else:
            noise_norm_function = lambda x: x * 0
        self.noise_head = NoiseHead(args.encoder_dim, dataset_utils.get_input_dim(args), noise_norm_function, args.n_mlp_blocks, args.setnet_beta)

    def forward(self, x, prev_pts_encoding):
        # Encoding
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                pts_encoding, sample_encoding = self.encoder(x, prev_pts_encoding)
        else:
            pts_encoding, sample_encoding = self.encoder(x, prev_pts_encoding)

        # Heads
        pred_outliers, pred_logits = self.class_head(pts_encoding)
        pred_shape = self.reg_head(sample_encoding)
        pred_features = self.features_head(pts_encoding)
        pred_noise = self.noise_head(pts_encoding)

        return pred_shape, pred_outliers, pred_logits, pred_noise, pred_features, pts_encoding

