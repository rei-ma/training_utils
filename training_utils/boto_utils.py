from typing import Dict, Tuple
import PIL
from PIL import Image
import boto3
import json
from botocore.errorfactory import ClientError
import logging
import os
from io import StringIO
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.DEBUG)))
logger.addHandler(logging.StreamHandler())


def read_image_from_s3(bucket_name: str, key: str) -> PIL.Image:
    """Read images from S3.

    Args:
        bucket_name: Bucket name in S3
        key: path to the object in S3

    Returns:
        im: PIL Image of a file in S3
    """
    try:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        object = bucket.Object(key)
        response = object.get()
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.error(f'No object found under {key} - returning empty')
            return None
        elif e.response['Error']['Code'] == 'NoSuchBucket':
            logger.error(f'Bucket ({bucket_name} not found - returning empty')
            return None
        else:
            raise

    file_stream = response['Body']
    im = Image.open(file_stream)
    return im


def get_json_document(path: str, bucket: str = "tinexx") -> Dict:
    """  Load a json document from the S3 and convert it to a dict

    Args:
        path: The path of the JSON Document
        bucket: The bucket where the path can be found

    Returns:
        Dict: Content of JSON document or None if not found
    """
    s3 = boto3.resource('s3')
    try:
        obj = s3.Object(bucket, path)
        str_rep = obj.get()['Body'].read().decode('utf-8')
        return json.loads(str_rep)
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.error(f'No object found under {path} - returning empty')
            return dict()
        elif e.response['Error']['Code'] == 'NoSuchBucket':
            logger.error(f'Bucket ({bucket} not found - returning empty')
            return dict()
        else:
            raise


def extract_text(ocr_path: str, bucket: str = "tinexx") -> str:
    """ Extract text from a file located in AWS S3

    Args:
        ocr_path: S3 Path of an OCR results file in the specified bucket
        bucket: S3 bucket containing the file

    Returns:
        str: OCR token separated by spaces
    """
    ocr = get_json_document(ocr_path, bucket)
    return extract_text_from_dict(ocr_dict=ocr)


def extract_text_from_dict(ocr_dict: Dict):
    """ Extract token from OCR dict and return string of all token

    1. Extract all token
    2. Replace whitespace with _
    3. Convert to lowercase
    4. Join together separating token by whitespace

    Args:
        ocr_dict: Doct containing all token (Textract result)

    Returns:
        str: All token separated by whitespace
    """
    text = []
    if ocr_dict is None or "Blocks" not in ocr_dict:
        return ""
    for token in ocr_dict["Blocks"]:
        if token["BlockType"] == "WORD":
            token_text = token["Text"]
            token_text = "_".join(token_text.split()) # join text that is seperated by a space in order to
                                                      # avoid issues with the combination of egnn and roberta
            text.append(token_text.lower())
    return " ".join(text)


def split_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """ Split S3 URI in bucket and s3 key

    Args:
        s3_uri: An S3 object Unique Resource Identifier

    Returns:
        str: bucket name
        str: s3 key
    """

    bucket = s3_uri.split("/")[2]
    s3_key = "/".join(s3_uri.split("/")[3:])

    return bucket, s3_key


def upload_dict(input_dict: Dict, s3_uri: str):
    """ Upload dict to S3

    Args:
        input_dict: Data to upload
        s3_uri: Path to upload file to

    Returns:
        None
    """
    bucket, s3_key = split_s3_uri(s3_uri)

    s3_resource = boto3.resource('s3')

    json_buffer = StringIO()
    json.dump(input_dict, json_buffer)
    s3_resource.Object(bucket, s3_key).put(Body=json_buffer.getvalue())


def read_s3_csv(s3_uri: str) -> pd.DataFrame:
    """ Read CSV file from s3 and return as pandas DataFrame

    Args:
        s3_uri: S3 URI of csv file

    Returns:
        pd.DataFrame: Dataframe containing the csv file values
    """
    s3_client = boto3.client("s3")
    bucket, key = split_s3_uri(s3_uri)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj.get("Body"))
    return df