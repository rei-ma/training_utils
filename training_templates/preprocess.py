"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
from typing import Dict

from training_utils.sagemaker_pipeline.preprocessor import SagemakerPreprocessor


logger = logging.getLogger(__name__)
logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.DEBUG)))
logger.addHandler(logging.StreamHandler())

class Preprocessor(SagemakerPreprocessor):
    def __init__(self, **kwargs):
        super(Preprocessor, self).__init__(**kwargs)
        # TODO: Add additional necessary arguments and class functions

    @staticmethod
    def transform_fn(data: Dict, transform_utilities: Dict) -> Dict:
        """ Transform a singe data point and return the transformed data

        This function will also be used in the inference script to prepare model data

        Args:
            data: Dictionary containing all necessary input data
            transform_utilities: Dictionary containing utility function, classes or information need for transformation

        Returns:
            Dict: Dict containing all transformed data
        """
        # TODO: Implement preprocessing logic for single data point
        # Adapt as necessary (input, output, ...) - keep changed signature in mind for batch_transform of this class
        # and input_fn in inference.py

        return {}

    def batch_transform(self, input_data_path: str, output_folder: str) -> None:
        """ Function to transform dataset and store it to disc

        1. Read input data
        2. Transform data and/or download additional data
        2.1 Use static transform_fn if no performance improvement can be achieved
            by transforming multiple data point at one
        3. save data in output folder

        Args:
            input_data_path: Path to local file
            output_folder: Path to local folder to store results

        Returns:
            None
        """
        # TODO: Implement transformation of full dataset; use transform_fn if it does not lead to major performance issues
        # Input signature cannot be changed!
        # Do not forget to store results and byproducts in output folder

        raise NotImplementedError("Batch tranformation function must be implemented")


def parse_args() -> argparse.Namespace:
    """ Read command line arguments

    Returns:
        argparse.Namespace: Namespace containing all arguments as class variables
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-train-data", type=str, required=True)
    parser.add_argument("--input-validation-data", type=str, required=True)
    parser.add_argument("--input-test-data", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default="/opt/ml/processing") # default value for sagemaker instance

    # TODO: Add arguments if necessary - don't forget to additional arguments in the pipeline as well
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")

    args = parse_args()

    processor = Preprocessor(
        training_input_data_path=args.input_train_data,
        validation_input_data_path=args.input_validation_data,
        test_input_data_path=args.input_test_data,
        base_dir=args.base_dir
    )
    processor.process()