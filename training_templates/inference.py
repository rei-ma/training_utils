import argparse

import torch
import json
import logging
from typing import Dict
import os


logger = logging.getLogger(__name__)
logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.DEBUG)))
logger.addHandler(logging.StreamHandler())


MODEL_NAME = "" # TODO: Change model name
LABEL_MAP = {} # TODO: Add mapping from int to label

OUT_CONTENT_TYPE = "application/json" # TODO: Check content type
IN_CONTENT_TYPE = "application/json" # TODO: Check content type

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def model_fn(model_dir: str, context=None):
    """Load the model for inference.
    
    Args:
        model_dir: directory of the model
        context: Sagemaker context
        
    Returns:
        model: loaded model
    """
    model = None # TODO: Load correct model
    model.to(device)

    return model

def input_fn(request_body, request_content_type, context=None):
    """ Deserialize and prepare the prediction input

    Args:
        request_body: Input str encoding data
        request_content_type: Typed of the request body
        context: Sagemaker context

    Returns:

    """

    if request_content_type == IN_CONTENT_TYPE:
        request = json.loads(request_body) # TODO: Adapt to content type

        preprocessed_data = None # TODO: Perform preprocessing in the same was as in preprocess.py

        return preprocessed_data
    else:
        logger.error(f"Encountered wrong input type. Got {request_content_type} but only accept {IN_CONTENT_TYPE}")
        return {}

def predict_fn(data: Dict, model: torch.nn.Module) -> Dict:
    """Calls a model on data deserialized in input_fn.
        Runs prediction on GPU if cuda is available.

    Args:
        data: Dictionary containing input data
        model: Trained torch module

    Returns:
        Dict: Output of the model (prediction, scores)
    """

    model.eval()

    # TODO: Adapt data extraction and model call
    output = model(data)

    prediction = output.argmax(dim=1).detach().cpu().numpy().item()

    predicted_class = LABEL_MAP[prediction]

    response = {
        "prediction": predicted_class,
        "scores": output.detach().cpu().numpy().tolist()
    }

    return response

def output_fn(prediction_output, accept=OUT_CONTENT_TYPE):
    """Serializes predictions from predict_fn to JSON

    If accept does not match OUT_CONTENT_TYPE the raw prediction output will be returned

    Args:
        prediction_output: Dict containing output from predict_fn
        accept: Request caller accept argument

    Returns:
        prediction output
    """

    if accept == OUT_CONTENT_TYPE:
        return json.dumps(prediction_output) # TODO: Adapt based on content type
    else:
        return prediction_output

def parse_args() -> argparse.Namespace:
    """ Read command line arguments for local test

    Returns:
        argparse.Namespace: Namespace containing all arguments as class variables
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model-path", type=str, required=True)
    parser.add_argument("--input-data-path", required=True,  action='append')
    parser.add_argument('--profiling', action='store_true')
    parser.add_argument('--no-profiling', dest='profiling', action='store_false')
    parser.set_defaults(profiling=False)

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    # for local tests

    args = parse_args()
    if args.profiling:
        from scalene import scalene_profiler

    model_path = args.input_model_path
    data_paths = args.input_data_path

    model = model_fn(model_path)
    for data_path in data_paths:
        with open(data_path) as f:
            request_dict = json.load(f)
        request_str = json.dumps(request_dict)

        if args.profiling:
            scalene_profiler.start()
        data = input_fn(request_str, request_content_type="application/json")

        out = predict_fn(data, model)

        final_output = output_fn(out)
        if args.profiling:
            scalene_profiler.stop()

        print(final_output)