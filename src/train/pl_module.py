# pylint: disable=arguments-differ,unused-argument,too-many-instance-attributes,too-many-arguments,line-too-long,comparison-with-itself
import warnings
import torch
import torch_geometric.data
import pytorch_lightning as pl

from src.model import InterfacedModel
from src.optim import OptimizerConfig, LRSchedulerConfig


class LitModule(pl.LightningModule):
    def __init__(
            self,
            model: InterfacedModel,
            sample_size: int,
            eval_sample_size: int,
            optimizer_config: OptimizerConfig,
            lr_scheduler_config: LRSchedulerConfig,
            verbose: bool=True
        ):
        super().__init__()
        self.model = model
        self.sample_size = sample_size
        self.eval_sample_size = eval_sample_size
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.lr_scheduler = None
        self.verbose = verbose

        self.criterion = model.criterion
        self.evaluator = model.evaluator
        self.metric_name = model.metric_name

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.predict_step_outputs = []

        self.save_hyperparameters(ignore=['model'])

    def configure_optimizers(self):
        optimizer = self.optimizer_config.setup(self.model)
        self.lr_scheduler = self.lr_scheduler_config.setup(optimizer)
        return optimizer

    def forward(self, x, n_samples=1):
        return self.model(x, n_samples)

    def parse_batch(self, batch):
        if isinstance(batch, torch_geometric.data.Data):
            if hasattr(batch, 'edge_label'):
                assert batch.y is None, 'Cannot have both y and edge_label'
                return batch, batch
            return batch, batch.y
        if isinstance(batch, (list, tuple)):
            return batch
        raise ValueError(f'Unknown batch type: {type(batch)}')

    def training_step(self, batch, batch_idx):
        assert self.model.training
        x, y = self.parse_batch(batch)
        y_hat, loss_dict = self(x, n_samples=self.sample_size)
        loss_total = 0

        loss_main = self.criterion(y_hat, y)
        loss_total = loss_total + (loss_main if torch.isfinite(loss_main) else 0)

        for loss_name in loss_dict.keys():
            loss_value = loss_dict[loss_name]['value']
            loss_total = loss_total + loss_dict[loss_name]['weight'] * loss_value
            self.log(f'training/loss_{loss_name}', loss_value, on_step=True, logger=True, sync_dist=True)

        self.lr_scheduler.step(self.global_step)
        self.log('training/loss_main', loss_main, on_step=True, logger=True, sync_dist=True)
        self.log('training/lr', self.lr_scheduler.lr, on_step=True, logger=True, sync_dist=True)
        self.log('step', float(self.global_step), on_step=True, logger=True, sync_dist=True)
        self.training_step_outputs.append((loss_main.detach(), loss_dict))
        return loss_total

    def on_train_epoch_end(self):
        if len(self.training_step_outputs) == 0:
            if self.verbose:
                warnings.warn("training_step_outputs is empty. This can happen when training is resumed from a checkpoint.")
            return

        losses_main, loss_dicts = zip(*self.training_step_outputs)

        epoch_loss_main = torch.stack([l for l in losses_main if l == l]).mean()
        self.log('training/loss_main_epoch', epoch_loss_main, on_epoch=True, logger=True, sync_dist=True)

        for loss_name in loss_dicts[0].keys():
            loss_epoch_mean = torch.stack([loss_dict[loss_name]['value'] for loss_dict in loss_dicts]).mean()
            self.log(f'training/loss_{loss_name}_epoch', loss_epoch_mean, on_epoch=True, logger=True, sync_dist=True)

        self.training_step_outputs.clear()

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def inference(self, x, n_samples=1):
        assert not self.model.training
        return self(x, n_samples)

    def validation_step(self, batch, batch_idx):
        x, y = self.parse_batch(batch)
        y_hat, loss_dict = self.inference(x, n_samples=self.eval_sample_size)

        val_output = (y_hat, y)

        val_loss_total = 0
        val_loss_main = self.criterion(y_hat, y)
        val_loss_total = val_loss_total + (val_loss_main if torch.isfinite(val_loss_main) else 0)

        for loss_name in loss_dict.keys():
            val_loss_total = val_loss_total + loss_dict[loss_name]['weight'] * loss_dict[loss_name]['value']

        self.validation_step_outputs.append((val_loss_main.detach(), val_output, loss_dict))
        return val_loss_total

    def on_validation_epoch_end(self):
        losses_main, outputs, loss_dicts = zip(*self.validation_step_outputs)

        epoch_loss_main = torch.stack([l for l in losses_main if l == l]).mean()
        self.log('validation/loss_main_epoch', epoch_loss_main, on_epoch=True, logger=True, sync_dist=True)

        perfs = self.evaluator(*zip(*outputs))

        if isinstance(self.metric_name, str):
            epoch_perf = perfs['metric_sum'] / perfs['metric_count']
            self.log(f'validation/{self.metric_name}_epoch', epoch_perf, on_epoch=True, logger=True, sync_dist=True)
        else:
            assert isinstance(self.metric_name, list)
            for metric_name in self.metric_name:
                epoch_perf = perfs[metric_name]['metric_sum'] / perfs[metric_name]['metric_count']
                self.log(f'validation/{metric_name}_epoch', epoch_perf, on_epoch=True, logger=True, sync_dist=True)

        for loss_name in loss_dicts[0].keys():
            losses = [loss_dict[loss_name]['value'] for loss_dict in loss_dicts]
            loss_epoch_mean = torch.stack(losses).mean()
            self.log(f'validation/loss_{loss_name}_epoch', loss_epoch_mean, on_epoch=True, logger=True, sync_dist=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = self.parse_batch(batch)
        y_hat, loss_dict = self.inference(x, n_samples=self.eval_sample_size)

        test_output = (y_hat, y)

        test_loss_total = 0
        test_loss_main = self.criterion(y_hat, y)
        test_loss_total = test_loss_total + (test_loss_main if torch.isfinite(test_loss_main) else 0)

        for loss_name in loss_dict.keys():
            test_loss_total = test_loss_total + loss_dict[loss_name]['weight'] * loss_dict[loss_name]['value']

        self.test_step_outputs.append((test_loss_main.detach(), test_output, loss_dict))
        return test_loss_total

    def on_test_epoch_end(self):
        losses_main, outputs, loss_dicts = zip(*self.test_step_outputs)

        epoch_loss_main = torch.stack([l for l in losses_main if l == l]).mean()
        self.log('test/loss_main_epoch', epoch_loss_main, on_epoch=True, logger=True, sync_dist=True)

        perfs = self.evaluator(*zip(*outputs))

        if isinstance(self.metric_name, str):
            epoch_perf = perfs['metric_sum'] / perfs['metric_count']
            self.log(f'test/{self.metric_name}_epoch', epoch_perf, on_epoch=True, logger=True, sync_dist=True)
        else:
            assert isinstance(self.metric_name, list)
            for metric_name in self.metric_name:
                epoch_perf = perfs[metric_name]['metric_sum'] / perfs[metric_name]['metric_count']
                self.log(f'test/{metric_name}_epoch', epoch_perf, on_epoch=True, logger=True, sync_dist=True)

        for loss_name in loss_dicts[0].keys():
            loss_epoch_mean = torch.stack([loss_dict[loss_name]['value'] for loss_dict in loss_dicts]).mean()
            self.log(f'test/loss_{loss_name}_epoch', loss_epoch_mean, on_epoch=True, logger=True, sync_dist=True)

        self.test_step_outputs.clear()
