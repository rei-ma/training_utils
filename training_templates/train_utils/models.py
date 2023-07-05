import torch
from torch import nn
import logging
import os


logger = logging.getLogger(__name__)
logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.DEBUG)))
logger.addHandler(logging.StreamHandler())


class MyModel(nn.Module):
    """Model class"""

    def __init__(self, trained_model_path: str = None, device: torch.device = torch.device("cpu")):
        """Initialization of model"""
        
        super(MyModel, self).__init__()
        self.name = ""
        # TODO: initialize class variables


    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass of model

        Args:
            data: input data to the model
            
        Returns:
            output: model output (prediction)
        """
        # TODO: perform forward pass
        return data
