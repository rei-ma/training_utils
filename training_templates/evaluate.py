"""Evaluation script for measuring mean squared error."""
import argparse
import json
import logging
import pathlib
import tarfile
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from typing import Tuple
from torch.utils.data import DataLoader
import os

from train_utils.dataset import MyDataset # TODO: change to correct class name
from train_utils.models import MyModel # TODO: change to correct class name

logger = logging.getLogger(__name__)
logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.DEBUG)))
logger.addHandler(logging.StreamHandler())

criterion = nn.CrossEntropyLoss() # TODO: add train class weights

BATCH_SIZE = 1

def predict_data(model: nn.Module, df_test: pd.DataFrame) -> Tuple[float, float]:
    """ Get prediction for dataset and calculate metrics

    Args:
        model: Torch Module to make predictions
        df_test: DataFrame containing information on data points

    Returns:
        float: Average accuracy on dataset
        float: Average loss on dataset
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = MyDataset(df_test) # TODO: change to correct class name
    test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=BATCH_SIZE)

    model.eval()

    labels = np.array([])
    preds = np.array([])
    loss = 0

    with torch.no_grad():
        for test_data, test_label in test_dataloader:
            test_label = test_label.to(device)

            # TODO: Adapt data extraction and model call
            output = model(test_data)
            prediction = output.argmax(dim=1).cpu().numpy()

            preds = np.append(preds, prediction)
            labels = np.append(labels, test_label.argmax(dim=1).cpu().numpy()) # TODO: change based on type of label

            loss += criterion(output, test_label.to(device)).detach().cpu().numpy()


    # TODO: Adapt metrics based on use-case
    avg_acc = np.sum(labels == preds) / len(preds)
    avg_loss = loss / len(preds)

    return avg_acc, avg_loss

def parse_args() -> argparse.Namespace:
    """ Read command line arguments

    Returns:
        argparse.Namespace: Namespace containing all arguments as class variables
    """
    parser = argparse.ArgumentParser() #TODO: Check default values
    parser.add_argument("--input-model-path", type=str, default="/opt/ml/processing/model/model.tar.gz")
    parser.add_argument("--input-data-path", type=str, default="/opt/ml/processing/test/dataset.csv")
    parser.add_argument("--output-path", type=str, default="/opt/ml/processing/evaluation")

    # TODO: Add arguments if necessary - don't forget to additional arguments in the pipeline as well
    args, _ = parser.parse_known_args()
    return args


def main():
    """ Evaluate model and save metrics to disc

    1. Load trained model
    2. Make predictions and calculate metrics
    3. Store metrics as json to disc

    Returns:
        None
    """

    args = parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    df_test = pd.read_csv(args.input_data_path)

    if args.input_model_path.endswith(".pt"):
        model_path = args.input_model_path
    elif args.input_model_path.endswith(".gz"):
        with tarfile.open(args.input_model_path) as tar:
            tar.extractall(path=".")
        model_path = "<robot>-<module>-<modelname>-<submodelname>.pt"
    else:
        raise ValueError(f"Unsupported format for model: {args.input_model_path.split('/')[-1]} \n"
                         f"Allowed formats: .pt, .tar.gz")

    model = MyModel()
    model.load_state_dict(torch.load(model_path, map_location=device))

    acc, loss = predict_data(model, df_test)

    report_dict = { # TODO: Change based on used metrics
        "classification_metrics": {
            "accuracy": {
                "value": acc,
            },
            "loss":{
                "value": loss
            }
        },
    }

    output_dir = args.output_path
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with accuracy: %f", acc)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))


if __name__ == "__main__":
    main()
