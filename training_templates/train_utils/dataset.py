from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.DEBUG)))
logger.addHandler(logging.StreamHandler())


class MyDataset(Dataset):
    """Sequence object to train a model on larger-than-memory data."""

    def __init__(self, df: pd.DataFrame):
        """Initialization of DataSequence

        Args:
            df: dataframe
        """
        # TODO: initialize class variables
        self.labels = []
        self.data = []

    def classes(self) -> List:
        """Return the labels of all of the data
        
        Returns:
            list: the labels of data
        """
        
        return self.labels
    
    
    def __len__(self) -> int:
        """Return the length of the data in each batch
        
        Returns:
            int: length of data
        """
        
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Fetch all the necessary data

        Args:
            idx: index of the data
        
        Returns:
            batch_encodings: batch encoding
            label: dataset label
        """
        # TODO: Change according to needs
        data = self.data[idx]
        label = self.labels[idx]

        return data, label
