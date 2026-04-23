# Final Discussion

## 1. Workflow Improvements

### Dataset Scaling

We increased the sample size from the "Appliance" category from 5000 to 10000 products. This allowed us to have a more comprehensive dataset for training and testing our RAG workflow, which can lead to improved performance and better generalization.

### LLM Experiment

We compared the performance of `Meta-Llama-3-8B-Instruct` with `...` using the same prompts and pipeline.

## 2. Additional Feature: Web Deployment

We deployed our web interface onto Posit Cloud.
  
## 3. Documentation and Code Quality Improvements

### Documentation Update

- Added instruction to use the web-deployed version.
- Added feature description for the web interface.
- Added usage examples.
-

### Code Quality Changes

- Added docstrings to all functions in the `src` directory, improving code readability and maintainability.
- Added a separate environment file `requirements.txt` for the web deployment. The dependencies in this file are specifically tailored for the web interface.
- Extracted the model training and building from `notebooks/milestone1_exploration.ipynb` into `src/build_artifacts.py`. This modularization allows reusability, making it easier to tune the training process and adjust sample size.

## 4: Cloud Deployment Plan
