# Training Utilities for Sagemaker Training

This repository provides utility functionality to train model using AWS Sagemaker

- General Utility function:
  - Transform images
  - Transform text
  - Handle data and interact with AWS S3 
- Sagemaker Training utility class for Preprocessing and Training

## Installation

### 1. Install PyTorch on your machine

```bash
export PYTORCH_VERSION=1.10.1
export TORCHVISION_VERSION=0.11.2
```

Cuda available: 
```bash
pip install torch==$PYTORCH_VERSION+cu111 torchvision==$TORCHVISION_VERSION+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Cuda not available: 
```bash
pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. Install this package

First you need to create a Project Access Token and store it in Environment variables

See https://docs.gitlab.com/ee/user/project/settings/project_access_tokens.html for more information on how to create a project access token.
