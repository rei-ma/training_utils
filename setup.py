import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='training_utils',
    version='1.0.0',
    description='Training utilities for model training using Sagemaker SDK',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls = {
        "Bug Tracker": "https://github.com/mike-huls/toolbox/issues"
    },
    packages=['training_utils', "training_utils.sagemaker_pipeline"],
    install_requires=[
        "nltk==3.7",
        "spacy==3.2.4",
        "numpy==1.22.4",
        "pillow==9.3.0",
        "boto3==1.26.10",
        "Pillow==9.3.0",
        "scikit-learn==1.0.2",
        "pandas==1.3.5"
    ]
)