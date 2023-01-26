import random

import numpy as np
import torch
from torchvision.utils import make_grid

from base.base_trainer import BaseTrainer
from utils.util import denormalize


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, config, model, loss, metrics, optimizer, lr_scheduler, resume, data_loader,
                 valid_data_loader=None, train_logger=None):
        super(Trainer, self).__init__(config, model, loss, metrics, optimizer, lr_scheduler, resume,
                                      train_logger)

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        # set models to train mode
        self.model.train()

        total_model_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, sample in enumerate(self.data_loader):
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

            # get data and send them to GPU
            input_image = sample['input_image'].to(self.device)
            ground_truth = sample['ground_truth'].to(self.device)

            # get G's output
            predicted = self.model(input_image)

            # denormalize
            with torch.no_grad():
                denormalized_input_image = denormalize(input_image)
                denormalized_ground_truth = denormalize(ground_truth)
                denormalized_predicted = denormalize(predicted)

            if batch_idx % 100 == 0:
                # save input_image, ground_truth and predicted image
                self.writer.add_image('input_image', make_grid(denormalized_input_image.cpu()))
                self.writer.add_image('ground_truth', make_grid(denormalized_ground_truth.cpu()))
                self.writer.add_image('predicted', make_grid(denormalized_predicted.cpu()))

            # train model
            self.model_optimizer.zero_grad()

            content_loss_lambda = self.config['others']['content_loss_lambda']
            content_loss_g = self.content_loss(predicted, ground_truth) * content_loss_lambda
            model_loss = content_loss_g

            self.writer.add_scalar('content_loss_g', content_loss_g.item())
            self.writer.add_scalar('model_loss', model_loss.item())

            model_loss.backward()
            self.model_optimizer.step()
            total_model_loss += model_loss.item()

            # calculate the metrics
            total_metrics += self._eval_metrics(denormalized_predicted, denormalized_ground_truth)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] model_loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        model_loss.item(),  # it's a tensor, so we call .item() method
                    )
                )

        log = {
            'model_loss': total_model_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        self.model_lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))

        with torch.no_grad():
            for batch_idx, sample in enumerate(self.valid_data_loader):
                input_image = sample['input_image'].to(self.device)
                ground_truth = sample['ground_truth'].to(self.device)

                predicted = self.model(input_image)

                content_loss_lambda = self.config['others']['content_loss_lambda']
                content_loss_g = self.content_loss(predicted, ground_truth) * content_loss_lambda
                loss_g = content_loss_g

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('content_loss_g', content_loss_g.item())
                self.writer.add_scalar('loss_g', loss_g.item())
                total_val_loss += loss_g.item()

                total_val_metrics += self._eval_metrics(denormalize(predicted), denormalize(ground_truth))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
