# Final Discussion

## 1. Workflow Improvements

### Dataset Scaling

We increased the sample size from the "Appliance" category from 5000 to 10000 products. This allowed us to have a more comprehensive dataset for training and testing our RAG workflow, which can lead to improved performance and better generalization.

### LLM Experiment

In `notebook/final_llm.ipynb`, we compared the performance of `Meta-Llama-3-8B-Instruct` with `Qwen2.5-7B-Instruct` on Hugging Face using the same prompts and pipeline.
    - **Model A**: meta-llama/Meta-Llama-3-8B-Instruct
    - **Model B**: Qwen/Qwen2.5-7B-Instruct

Both Llama 3 and Qwen 2.5 are instruction-tuned, general-purpose language models with similar context windows. [Results](https://llm-stats.com/models/compare/llama-3.1-8b-instruct-vs-qwen-2.5-7b-instruct) indicate that Qwen 2.5 shows notably better performance in the majority of benchmarks.

Initially, `mistralai/Mistral-7B-Instruct-v0.3` was considered as a strong candidate, but the current Hugging Face API integration in our project uses a chat-completion interface, and this model was not supported through that route. Considering code reusability and ease of integration, we only compare Llama 3 and Qwen 2.5.

The below prompts were used:

```txt
You are a helpful Amazon shopping assistant.
Answer using ONLY the provided context. Do not use outside knowledge.
Write 2-4 bullet points covering:
- best match(es)
- why they are relevant
- any important limitation or uncertainty
If the retrieved context does not clearly support an answer, explicitly say that.
```

All outputs from both models are documented in `notebook/final_llm.ipynb`. Both models were able to provide natural-language responses and user-friendly formatting. However, Qwen demonstrated superior "intelligence", such as realizing that a mini-fridge is a poor recommendation for a family of four, despite the context describing it as "large capacity" (relative to other mini-fridges). Also, Llama had a tendency to include accessories (power cords, gap covers) as "best matches" for appliance queries. This would be frustrating for a user looking for an actual machine. We decided to use Qwen 2.5 as our default model for the web interface.

## 2. Additional Feature: Web Deployment

We deployed our web interface onto Posit Connect Cloud. The app can now be access at <https://yhouyang02-dsci-575-project-yuhengo-mkcchoy.share.connect.posit.cloud>. Due to the limitations of the free-tier plan, the build and loading time on this public cloud can take up to 10 minutes. To optimize the deployment, we would recommend using a scalable cloud computing environment such as AWS EC2, which can provide more resources and faster loading times.
  
## 3. Documentation and Code Quality Improvements

### Documentation Update

- Added instruction to use the web-deployed version.
- Added instruction for developers to rebuild the model artifacts.
- Added instruction for developers to build and run the app locally.

### Code Quality Changes

- Added docstrings to all functions in the `src` directory, improving code readability and maintainability.
- Added short descriptions for all `.py` files in the `src` directory.
- Added a separate environment file `requirements.txt` for the web deployment.
- Extracted the model training and building from `notebooks/milestone1_exploration.ipynb` into `src/build_artifacts.py`. This modularization allows reusability, making it easier to tune the training process and adjust sample size.

## 4: Cloud Deployment Plan

- **Data Storage**: To deploy our Amazon product recommendation tool on a cloud platform such as AWS, we would rely on a combination of managed storage, scalable compute, and automated update pipelines. All raw product data would be stored in Amazon S3. After preprocessing, the cleaned datasets would also be saved in S3. Both the BM25 keyword index and the vector index used for semantic search would be hosted in Amazon OpenSearch Serverless with its vector engine enabled.

- **Compute**: The application would run either as a serverless API on AWS Lambda. Lambda is sufficient for lightweight, stateless requests and automatically scales to handle concurrent users. User traffic would be routed through Amazon API Gateway, which manages concurrency and request throttling. For LLM inference, we would rely on an external API such as OpenAI rather than hosting our own model.

- **Streaming/Updates**: To keep the system up to date with new Amazon products, we would schedule a recurring ingestion job using Amazon EventBridge. Each run would fetch new product data, store it in S3, and trigger a function that cleans the data, generates embeddings, and updates both the vector index and the BM25 index. This ensures that the recommendation system continuously incorporates new items without manual intervention. CloudWatch monitoring would track failures or delays in the pipeline so that issues can be addressed quickly.
