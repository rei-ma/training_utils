import os.path
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.utils.data import Dataset
from argparse import Namespace, ArgumentParser
import logging
from typing import Dict, Tuple
from sklearn.utils import class_weight
import numpy as np

from training_utils.sagemaker_pipeline.trainer import SagemakerDLTrainer
from train_utils.models import MyModel
from train_utils.dataset import MyDataset


logger = logging.getLogger(__name__)
logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.DEBUG)))
logger.addHandler(logging.StreamHandler())


class Trainer(SagemakerDLTrainer):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)

    @staticmethod
    def forward_fn(model: nn.Module, data: Dict, device: torch.device) -> torch.Tensor:
        """ Transform and extract that so that it can be fed into the model class

        1. Extract necessary torch.Tensors from data object
        2. Move Tensors to same device as model
        3. (optional) Add/Remove dimensions as necessary based on batch size
        4. Feed tensors in model
        5. (optional) Postprocess the output to be used in the criterion

        Args:
            model: Torch module to transform data and save gradients
            data: Input data dictionary for the model
            device: Device model is stored on

        Returns:
            torch.Tensor: result of prediction
        """
        raise NotImplementedError("Model function not implemented by subclass")


def model_fn(device: torch.device) -> nn.Module:
    """ Initialize model class and move to device

    Args:
        device: torch device to store model

    Returns:
        torch.Module: Model
    """
    model = MyModel() # TODO: Change to implemented model name
    model = model.to(device)
    raise NotImplementedError("Model initialisation not implemented")


def get_criterion(label_weight: torch.Tensor) -> nn.Module:
    """ Initialize the criterion used to evaluate the model

    Args:
        label_weight: Class weights for each class

    Returns:
        torch.Module: Loss function implemented as torch.Module (=criterion)
    """
    # TODO: Change criterion if necessary
    criterion = nn.CrossEntropyLoss(weight=label_weight)
    return criterion


def get_optimizer(model: nn.Module, learning_rate: float) -> Optimizer:
    """  Initialize the optimizer used to update the model weights based on the gradients

    Args:
        model: Model to be updated
        learning_rate: Learning rate used

    Returns:
        Optimizer: A torch optimizer class instance
    """
    # TODO: Change optimizer if necessary
    optimizer = Adam(model.parameters(), lr=learning_rate)
    return optimizer


def load_datasets(train_data_path: str, validation_data_path: str) -> Tuple[Dataset, Dataset]:
    """ Load and return the dataset instances for training and validation

    Initialize both training and validation dataset class

    Args:
        train_data_path: Path to training data folder
        validation_data_path: Path to validation data folder

    Returns:
        Dataset: torch dataset containing the training data
        Dataset: torch dataset containing the validation data
    """
    train_dataset = MyDataset()
    validation_dataset = MyDataset()
    raise NotImplementedError("Dataset initialisation not implemented")


def compute_weights(dataset) -> torch.Tensor:
    """Compute the weight distribution according to the number of occurrence of each label
       and assign the weight to each sample

    Args:
        dataset: torch dataset containing training data

    Returns:
        class_weights: weights for each class
    """

    # TODO: Remove function if not needed
    my_label = dataset.classes()

    class_weights = class_weight.compute_class_weight(
        class_weight = 'balanced',
        classes = np.unique(my_label),
        y = my_label)

    return torch.tensor(class_weights, dtype=torch.float32)


def parse_args() -> Namespace:
    """ Read command line arguments

    Returns:
        argparse.Namespace: Namespace containing all arguments as class variables
    """
    parser = ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=os.environ.get('SM_NUM_CPUS'))
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false')
    parser.add_argument('--use_tqdm', action='store_true')
    parser.set_defaults(wandb=True, tqdm=False)

    # input data and model directories
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--checkpoint_dir', type=str, default="/opt/ml/checkpoints/")
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    args, _ = parser.parse_known_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset, validation_dataset = load_datasets(args.train, args.validation)

    class_weight = compute_weights(train_dataset).to(device)
    criterion = get_criterion(class_weight)

    trainer = Trainer(
        criterion=criterion,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        use_cuda=True,
        use_wandb=args.wandb,
        model_dir=args.model_dir,
        checkpoint_dir=args.checkpoint_dir,
        use_tqdm=args.use_tqdm
    )

    model = model_fn(device)
    optimizer = get_optimizer(model, args.learning_rate)

    trainer.train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset
    )
