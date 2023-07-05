"""Example workflow pipeline script for abalone pipeline.

                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os
import json
import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CacheConfig
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession
from typing import List
from sagemaker.workflow.retry import (
    SageMakerJobExceptionTypeEnum,
    SageMakerJobStepRetryPolicy
)

cache_config = CacheConfig(enable_caching=True, expire_after="P60d")

resource_limit_retry_policy = SageMakerJobStepRetryPolicy(
    failure_reason_types=[
        SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT,
    ],
    max_attempts=5,
    interval_seconds=600,
    backoff_rate=5
)


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region: str) -> sagemaker.session.Session:
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region: str, default_bucket: str, pipeline_mode: str) -> sagemaker.session.Session:
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts
        pipeline_mode: Execution mode for sagemaker pipeline (local_mode / sagemaker)

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    if pipeline_mode == "local_mode":
         return LocalPipelineSession(
            boto_session=boto_session,
            default_bucket=default_bucket,
        )
    elif pipeline_mode == "sagemaker":
        sagemaker_client = boto_session.client("sagemaker")
        return PipelineSession(
            boto_session=boto_session,
            sagemaker_client=sagemaker_client,
            default_bucket=default_bucket,
        )


def get_pipeline_custom_tags(new_tags: List, region: str, sagemaker_project_arn: str = None) -> List:
    """ Get SM project tags to add to pipeline in order to link them

    Args:
        new_tags: Any already defined tags
        region: AWS region
        sagemaker_project_arn: Sagemaker Project ARN

    Returns:
        List: List of all tags
    """
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    role=None,
    default_bucket="tinexx",
    model_package_group_name="<robot>-<component>-<model>",
    pipeline_name="training-<robot>-<component>-<model>",
    base_job_prefix="training-<robot>-<component>-<model>",
    pipeline_mode="sagemaker",
    sample_train_data="",
    sample_validation_data="",
    sample_test_data="",
    train_data="",
    validation_data="",
    test_data=""
):
    """Gets a SageMaker ML Pipeline instance

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts
        model_package_group_name: The name of the final model
        pipeline_name: The name of the pipeline
        base_job_prefix: The prefix for all jobs started through this pipeline
        pipeline_mode: Execution mode of sagemaker pipeline (local_mode / sagemaker)
        sample_train_data: S3 URI for sample training data for local testing
        sample_validation_data: S3 URI for sample validation data for local testing
        sample_test_data: S3 URI for sample test data for local testing
        train_data: S3 URI for train data
        validation_data: S3 URI for validation data
        test_data: S3 URI for test data
    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket, pipeline_mode)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.g4dn.2xlarge"
    )
    # TODO: Addd different training instances for different submodels if needed
    #  (also add to pipeline-arguments in initiation of Pipeline)

    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    input_train_data = ParameterString(
        name="TrainInputDataURI",
        default_value=train_data
    )

    input_validation_data = ParameterString(
        name="ValidationInputDataURI",
        default_value=validation_data
    )

    input_test_data = ParameterString(
        name="TestInputDataURI",
        default_value=test_data
    )

    if pipeline_mode == "local_mode":  # use only small dataset
        input_train_data.default_value = sample_train_data
        input_validation_data.default_value = sample_validation_data
        input_test_data.default_value = sample_test_data
        processing_instance_type.default_value = "local_cpu"
        training_instance_type.default_value = "local_gpu"
        train_parameters = {"num_workers": 0, "batch_size": 1, "epochs": 1, "no-wandb": ""}
    else:
        train_parameters = {}

    with open(os.path.join(BASE_DIR, "model_metadata.json")) as f:
        model_metadata = json.load(f)

    # TODO: Define image uris to use for preprocessing, training, evaluation and inference
    preprocess_image_uri = ""
    train_image_uri = ""
    evaluate_image_uri = ""
    inference_image_uri = ""

    # processing step for feature engineering
    preprocessor = ScriptProcessor(
        image_uri=preprocess_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/preprocess",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    step_process = ProcessingStep(
        name="", # TODO: Name preprocessing step
        processor=preprocessor,
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=["--input-train-data",
                       input_train_data,
                       "--input-validation-data",
                       input_validation_data,
                       "--input-test-data",
                       input_test_data],
        cache_config=cache_config
    )

    # training step for generating model artifacts
    model_path= f"s3://{sagemaker_session.default_bucket()}/Models/{model_package_group_name}"

    train = Estimator(
        image_uri=train_image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/train",
        sagemaker_session=sagemaker_session,
        role=role,
        entry_point="train.py",
        source_dir=BASE_DIR,
        hyperparameters=train_parameters if pipeline_mode == "local_mode" else None
    )

    step_train = TrainingStep(
        name="", # TODO: Name training step
        estimator=train,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="", # TODO: define input type of data
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="", # TODO: define input type of data
            ),
        },
        cache_config=cache_config,
        retry_policies=[resource_limit_retry_policy]

    )

    # processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=evaluate_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    evaluation_report = PropertyFile(
        name="", # TODO: define name of results report
        output_name="evaluation",
        path="evaluation.json",
    )

    step_eval = ProcessingStep(
        name="", # TODO: Name evaluation step
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
            ProcessingInput(
                source=os.path.join(BASE_DIR, "train_utils"),
                destination="/opt/ml/processing/input/code/train_utils",
            )
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
    )

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )

    model = Model(
        name=model_package_group_name,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        entry_point="inference.py",
        source_dir=BASE_DIR,
        image_uri=train.training_image_uri(),
        sagemaker_session=sagemaker_session
    )

    if pipeline_mode == "local_mode":
        model_step_args = model.create(
            instance_type="ml.m5.xlarge",
        )
    else:
        model_step_args = model.register(
            content_types=[],  # TODO: Define input content type
            response_types=[],  # TODO: Define output content type
            model_package_group_name=model_package_group_name,
            image_uri=inference_image_uri,
            model_metrics=model_metrics,
            approval_status=model_approval_status,
            description="", # TODO: Describe model use-case
            customer_metadata_properties=model_metadata
        )

    model_step = ModelStep(
        name="RegisterModel",
        step_args=model_step_args
    )

    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step=step_eval,
            property_file=evaluation_report,
            json_path="classification_metrics.accuracy.value"
        ),
        right=0.0, # TODO: Set condition to load model to registry
    )
    step_cond = ConditionStep(
        name="", # TODO: Name condition step
        conditions=[cond_lte],
        if_steps=[model_step],
        else_steps=[],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_train_data,
            input_validation_data,
            input_test_data
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
