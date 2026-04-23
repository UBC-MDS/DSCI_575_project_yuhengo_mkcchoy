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

- Data Storage: To deploy our Amazon product recommendation tool on a cloud platform such as AWS, we would rely on a combination of managed storage, scalable compute, and automated update pipelines. All raw product data would be stored in Amazon S3. After preprocessing, the cleaned datasets would also be saved in S3. The vector index used for semantic search would be hosted in Amazon OpenSearch Serverless with its vector engine enabled, while the BM25 keyword index would be maintained in the same OpenSearch environment.

- Compute: The application would run either as a serverless API on AWS Lambda. Lambda is sufficient for lightweight, stateless requests and automatically scales to handle concurrent users. User traffic would be routed through Amazon API Gateway, which manages concurrency and request throttling. For LLM inference, we would rely on an external API such as OpenAI rather than hosting our own model.

- Streaming/Updates: To keep the system up to date with new Amazon products, we would schedule a recurring ingestion job using Amazon EventBridge. Each run would fetch new product data, store it in S3, and trigger a function that cleans the data, generates embeddings, and updates both the vector index and the BM25 index. This ensures that the recommendation system continuously incorporates new items without manual intervention. CloudWatch monitoring would track failures or delays in the pipeline so that issues can be addressed quickly.
