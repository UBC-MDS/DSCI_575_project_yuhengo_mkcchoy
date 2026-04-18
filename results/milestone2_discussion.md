# Qualitative evaluation for hybrid RAG search

## Model choice and rationale

We use Meta-Llama-3-8B-Instruct through the Hugging Face Inference API because it is a strong general-purpose instruction model and using an API avoids the burden of hosting an LLM locally.

## Results

We run the same queries as in `results/milestone1` and compare results of 5 out of 10 queries. The results (with a text limit) are recorded in `notebooks/milestone2_rag.ipynb`. We evaluate the results based on three dimensions (accuracy completeness, and fluency). Due to the nature of LLMs, the responses vary on each run, so the evaluation results can change if you run the test multiple times. Therefore, this is only a rough glance of the performance of the RAG-driven result. To carefully evaluate the model's performance, we will require a more thoughtful design of the qualitative rubrics and perform a large-scale analysis.

| Query | Accuracy | Completeness | Fluency | Brief Notes |
| --- | --- | --- | --- | --- |
| `gas range 30 inch` | Yes | Yes | Yes | The answer directly addresses the request by listing relevant 30-inch gas ranges. |
| `something to keep drinks cold in a dorm room` | Yes | Yes | Yes | The answer identifies relevant mini fridge options and matches the use case well. |
| `appliance for washing dishes quietly at night` | Yes | Yes | Yes | The response addresses the goal by identifying a dishwasher described as quiet. This is an example of retrieval form semantic queries. |
| `best dishwasher for a small apartment under $800` | No | No | Yes | The answer gives a specific recommendation, but it does not justify why it is the “best” option. The wording is fluent, but the evidence is incomplete. |
| `reliable stove for frequent home cooking that is easy to clean` | No | No | Yes | The answer shifts from recommending a stove to recommending burner covers. It is readable, but does not fullfil the goal. |

### Key observations

The Hybrid RAG workflow performs well on straightforward and semantic queries, especially when the retrieved context contains clear product matches such as "compact refrigerators", "quiet dishwashers", or "30-inch gas ranges". Its performance drops on more complex recommendation-style queries that involve multiple constraints such as price, reliability, or family size.

Given the limited training size and fine-tuning of the prompts, this level of performance is acceptable and proves that the workflow can already support basic product assistance tasks.

### Limitations of the hybrid RAG workflow

- The workflow does not reliably handle queries with multiple constraints, especially when the retrieved context does not explicitly mention all aspects of the request. This sometimes leads to weak recommendations.
- The generator can overly rely on loosely related items and produces answers that are fluent but not fully aligned with the user’s intent. This is visible when the system recommends accessories or related products instead of the core product requested.

### Suggestions for improving the performance

- Strengthening retrieval quality with better reranking or filtering after the hybrid retriever. For example, considering product type, price mentions, and key attributes could help ensure that the final context better matches the query's goal.
- Making the prompt more aware of the constraints, such as requiring the model to explicitly check whether each part of the question is supported by the retrieved context.
