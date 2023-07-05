import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _collate_fn_t
import torch.nn as nn
from tqdm import tqdm
import os
from pathlib import Path
import shutil
from typing import Dict, Tuple, Optional, List
import wandb


class SagemakerDLTrainer:
    """Class providing necessary training functionality for deep learning models
    Implements training loop with validation and early stopping
    """
    def __init__(self,
                 criterion,
                 num_workers: int = os.environ.get('SM_NUM_CPUS'),
                 batch_size: int = 1,
                 epochs: int = 100,
                 patience: int = 10,
                 learning_rate: float = 0.001,
                 use_cuda: bool = True,
                 use_wandb: bool = False,
                 model_dir: str = os.environ.get('SM_MODEL_DIR'),
                 checkpoint_dir: str = "/opt/ml/checkpoints/",
                 use_tqdm: bool = False,
                 use_accuracy: bool = True,
                 dataloader_collate_fn: Optional[_collate_fn_t] = None,
                 prediction_threshold: Optional[float] = None,
                 learning_rate_decay_ratios: Optional[List[float]] = None
    ):
        """ Create a SagemakerDLTrainer instance

        Args:
            criterion: Criterion function to measure performance
            num_workers: Number of workers for torch dataloader
            batch_size: Data batch size during training
            epochs: Maximal number of epochs to train model
            patience: Number of epochs without validation performance increase before early stopping training
            use_cuda: Indicator if cuda should be used if available
            model_dir: Path to folder to store model weights
            checkpoint_dir: Path to folder to store model weights during training
            use_tqdm: Indicator if tqdm should be used
            use_accuracy: Indicator if accuracy should be computed
            dataloader_collate_fn: Collate function used in torch dataloader
            prediction_threshold: Optional prediction threshold for models returning Tensor of Shape (n,)
            learning_rate_decay_ratios: List of decay ratios to apply after patience hits 0 during training
        """
        self.logger = logging.getLogger("Trainer")
        self.logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.DEBUG)))
        self.logger.addHandler(logging.StreamHandler())

        use_cuda = torch.cuda.is_available() and use_cuda
        device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device
        self.logger.debug(f"GPU used: {use_cuda}")

        self.criterion = criterion
        self.dataloader_collate_fn = dataloader_collate_fn
        self.prediction_threshold = prediction_threshold
        self.use_accuracy = use_accuracy

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.learning_rate_decay_ratios = learning_rate_decay_ratios

        self.use_tqdm = use_tqdm
        self.use_wandb = use_wandb

        self.model_dir = model_dir
        self.checkpoint_dir = checkpoint_dir

    def configure_wandb(self, model_name: str):
        """ Configure a Weights and Biases run

        Returns:

        """
        # use model as project name and not submodel
        if model_name.count("-") >= 3:
            project_name = "-".join(model_name.split("-")[:-1])
        else:
            project_name = model_name

        if self.use_wandb:
            run = wandb.init(project=project_name)

            config = wandb.config

        else:
            run = None
            config = type('', (), {})()

        config.batch_size = self.batch_size
        config.max_epochs = self.epochs
        config.learning_rate = self.learning_rate
        config.patience = self.patience

    def training_epoch(self, model: nn.Module, optimizer, dataloader: DataLoader) -> Tuple[float, float]:
        """ Training model for one epoch with provided dataloader

        Calculate training loss and accuracy and update model weights based on criterion and optimizer

        Args:
            model: Torch module implementing deep learning model
            optimizer: Optimizer used to update model weights
            dataloader: Torch dataloader providing data

        Returns:
            float: Average accuracy during training
            float: Average loss during training
        """
        total_correct = 0
        total_loss = 0
        total_data_points = 0

        model.train()

        for train_data, train_label in tqdm(dataloader, disable=not self.use_tqdm):
            model.zero_grad()

            output = self.forward_fn(model, train_data, self.device)
            if isinstance(output, Tuple) and isinstance(train_label, Tuple):
                batch_loss = self.criterion(*output, *all_tensors_to_device(train_label, self.device))
            elif isinstance(output, Tuple):
                train_label = train_label.to(self.device)
                batch_loss = self.criterion(*output, train_label)
            elif isinstance(train_label, Tuple):
                batch_loss = self.criterion(output, *all_tensors_to_device(train_label, self.device))
            else:
                train_label = train_label.to(self.device)
                batch_loss = self.criterion(output, train_label)

            if torch.isinf(batch_loss):
                self.logger.warning("Infinite loss encountered - used for backpropagation")
                batch_loss.backward()
            elif torch.isnan(batch_loss):
                self.logger.warning("NaN loss encountered - not used for backpropagation")
            else:
                batch_loss.backward()
                total_loss += batch_loss.item()

            if self.use_accuracy:
                train_label = train_label.argmax(dim=1) if len(train_label.shape) > 1 else train_label
                if self.prediction_threshold is not None:
                    prediction = (output >= self.prediction_threshold).long()
                else:
                    prediction = output.argmax(dim=1)

                num_correct = (prediction == train_label).sum().item()

                total_correct += num_correct
            else:
                total_correct = 0

            if isinstance(output, Tuple):
                total_data_points += output[0].shape[0]
            else:
                total_data_points += output.shape[0]

            optimizer.step()

        train_acc = total_correct / total_data_points if self.use_accuracy else None
        train_loss = total_loss / len(dataloader)

        return train_acc, train_loss

    def validation_epoch(self, model: nn.Module, dataloader: DataLoader):
        """ Validate model performance on with dataloader

        Get all model predictions and calculate loss and accuracy
        Model weights not updated in this step

        Args:
            model: Torch module implementing deep learning model
            dataloader: Torch dataloader providing data

        Returns:
            float: Average accuracy during validation
            float: Average loss during validation
        """
        total_loss = 0
        total_correct = 0
        total_data_points = 0

        model.eval()

        with torch.no_grad():

            for val_data, val_label in tqdm(dataloader, disable=not self.use_tqdm):
                output = self.forward_fn(model, val_data, self.device)

                if isinstance(output, Tuple) and isinstance(val_label, Tuple):
                    batch_loss = self.criterion(*output, *all_tensors_to_device(val_label, self.device))
                elif isinstance(output, Tuple):
                    val_label = val_label.to(self.device)
                    batch_loss = self.criterion(*output, val_label)
                elif isinstance(val_label, Tuple):
                    batch_loss = self.criterion(output, *all_tensors_to_device(val_label, self.device))
                else:
                    val_label = val_label.to(self.device)
                    batch_loss = self.criterion(output, val_label)

                total_loss += batch_loss.item()

                if self.use_accuracy:
                    val_label = val_label.argmax(dim=1) if len(val_label.shape) > 1 else val_label
                    if self.prediction_threshold is not None:
                        prediction = (output >= self.prediction_threshold).long()
                    else:
                        prediction = output.argmax(dim=1)

                    num_correct = (prediction == val_label).sum().item()
                    total_correct += num_correct

                if isinstance(output, Tuple):
                    total_data_points += output[0].shape[0]
                else:
                    total_data_points += output.shape[0]

        val_accuracy = total_correct / total_data_points
        val_loss = total_loss / len(dataloader)

        return val_accuracy, val_loss

    def train(self, model: nn.Module, optimizer, train_dataset: Dataset, validation_dataset: Dataset):
        """ Full training loop for deep learning model

        1. Initialize Dataloader based on dataset
        2. Training Loop using train_epoch and validataion_epoch
        3. Early stopping
        4. Store model weights in model folder and model checkpoints in checkpoint folder

        Args:
            model: Torch module implementing deep learning model
            optimizer: Optimizer used to update model weights
            train_dataset: Torch dataset providing data for training
            validation_dataset: Torch dataset providing data for validation

        Returns:
            None
        """
        self.logger.debug("Starting training ...")
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        self.configure_wandb(model.name)

        train_dataloader = DataLoader(train_dataset, num_workers=self.num_workers, batch_size=self.batch_size,
                                      shuffle=True, collate_fn=self.dataloader_collate_fn)
        val_dataloader = DataLoader(validation_dataset, num_workers=self.num_workers, batch_size=self.batch_size,
                                    collate_fn=self.dataloader_collate_fn)

        best_acc = 0
        best_loss = np.inf
        best_epoch = 0
        patience = self.patience
        decay_executed = 0

        for epoch_num in range(self.epochs):
            train_acc, train_loss = self.training_epoch(model, optimizer, train_dataloader)
            val_acc, val_loss = self.validation_epoch(model, val_dataloader)

            if val_loss <= best_loss:
                best_acc = val_acc
                best_loss = val_loss
                torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, model.name + ".pt"))
                patience = self.patience
                best_epoch = epoch_num
            else:
                patience -= 1

            log = {
                "Epochs": epoch_num,
                "Train-Loss": train_loss,
                "Train-Accuracy": train_acc,
                "Val-Loss": val_loss,
                "Val-Accuracy": val_acc,
            }

            self.logger.info(" |".join([str(key) + ": " + f"{value: .3f}" for key, value in log.items() if value is not None]) + f"  | Patience: {patience}")
            if self.use_wandb:
                wandb.log(log)

            if patience <= 0:
                if self.learning_rate_decay_ratios is None or decay_executed >= len(self.learning_rate_decay_ratios):
                    break
                self.logger.info(f"Patience exhausted: Decaying learning rate "
                             f"by {self.learning_rate_decay_ratios[decay_executed]} and resetting patience")
                adjust_learning_rate(optimizer, self.learning_rate_decay_ratios[decay_executed])
                patience = self.patience
                decay_executed += 1

        self.logger.info(f"Finished training with best model after epoch {best_epoch} with "
                         f"Val-Acc: {best_acc: .3f} and "
                         f"Val-Loss: {best_loss: .3f}")

        self.logger.debug("Moving model to final model directory...")
        shutil.copyfile(os.path.join(self.checkpoint_dir, model.name + ".pt"),
                        os.path.join(self.model_dir, model.name + ".pt"))

    @staticmethod
    def forward_fn(model: nn.Module, data: Dict, device: torch.device) -> torch.Tensor:
        """ Transform and extract that so that it can be fed into the model class

        Eg. Move input tensors to same device as model
            Add/Remove dimensions as necessary based on batch size

        Args:
            model: Torch module to transform data and save gradients
            data: Input data dictionary for the model
            device: Device model is stored on

        Returns:
            torch.Tensor: result of prediction
        """
        raise NotImplementedError("Model function not implemented by subclass")


def adjust_learning_rate(optimizer: torch.optim, scale: float):
    """
    Scale learning rate by a specified factor.

    Args:
        optimizer: optimizer whose learning rate must be shrunk.
        scale: factor to multiply learning rate with.

    Returns:
        None
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale


def all_tensors_to_device(nested_list: List, device: torch.device) -> List:
    """ Move all tensors in the nested list to the given device and return list

    Args:
        nested_list: Arbitrarily nested list containing torch.Tensors as elements
        device: Device to move the tensors to

    Returns:
        List: same structure as input list but with tensors on device
    """

    new_list = []

    for x in nested_list:
        if isinstance(x, List) or isinstance(x, Tuple):
            new_list.append(all_tensors_to_device(x, device))
        else:
            new_list.append(x.to(device))

    return new_list
