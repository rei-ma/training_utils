import shutil
from typing import Dict
import logging
import os
import pathlib
import boto3

class SagemakerPreprocessor:
    def __init__(self,
                 training_input_data_path: str,
                 validation_input_data_path: str,
                 test_input_data_path: str,
                 base_dir:str = "/opt/ml/processing"
                 ):
        """ Create SagemakerPreprocessor instance

        Args:
            training_input_data_path: S3 Uri to training dataset (csv)
            validation_input_data_path: S3 Uri to validation dataset (csv)
            test_input_data_path: S3 Uri to test dataset (csv)
            base_dir: Path to local directory to store raw and processed data
        """
        self.logger = logging.getLogger("Preprocessor")
        self.logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.DEBUG)))
        self.logger.addHandler(logging.StreamHandler())

        self.training_input_data_path = training_input_data_path
        self.validation_input_data_path = validation_input_data_path
        self.test_input_data_path = test_input_data_path

        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data")
        pathlib.Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        self.boto_s3 = boto3.resource("s3")


    def download_file(self, s3_filepath: str):
        """ Download file from S3 and save to data_dir

        Use same filename for local file as remote file

        Args:
            s3_filepath:

        Returns:

        """
        input_data = s3_filepath
        bucket = input_data.split("/")[2]
        s3_key = "/".join(input_data.split("/")[3:])

        filename = s3_key.split("/")[-1]
        local_path = os.path.join(self.data_dir, filename)

        self.logger.debug(f"Downloading data from bucket: {bucket}, key: {s3_key} to local path: {local_path}")
        self.boto_s3.Bucket(bucket).download_file(s3_key, local_path)

        return local_path

    def process(self):
        """ Process all three datasets using the batch_transform function

        1. Download dataset files
        2. Create all necessary files
        3. Perform data transformation and store to in data_dir

        Returns:
            None
        """
        self.logger.info("Downloading all input files...")
        if "s3" in self.training_input_data_path:
            train_path = self.download_file(self.training_input_data_path)
        else:
            train_path = self.training_input_data_path
        if "s3" in self.validation_input_data_path:
            validation_path = self.download_file(self.validation_input_data_path)
        else:
            validation_path = self.validation_input_data_path
        if "s3" in self.test_input_data_path:
            test_path = self.download_file(self.test_input_data_path)
        else:
            test_path = self.test_input_data_path


        pathlib.Path(f"{self.base_dir}/train").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{self.base_dir}/validation").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{self.base_dir}/test").mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Transforming data and writing to {self.base_dir}")
        self.batch_transform(train_path, os.path.join(self.base_dir, "train"))
        self.batch_transform(validation_path, os.path.join(self.base_dir, "validation"))
        self.batch_transform(test_path, os.path.join(self.base_dir, "test"))

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
        raise NotImplementedError("Batch data transformation function not implemented by subclass")

    @staticmethod
    def transform_fn(data: Dict, transform_utilities: Dict = None) -> Dict:
        """ Transform a singe data point and return the transformed data

        This function will also be used in the inference script to prepare model data

        Args:
            data: Dictionary containing all necessary input data
            transform_utilities: Dictionary containing utility function, classes or information need for transformation

        Returns:
            Dict: Dict containing all transformed data
        """
        raise NotImplementedError
