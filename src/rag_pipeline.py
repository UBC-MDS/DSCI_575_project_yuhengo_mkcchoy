"""
RAG pipeline with semantic, BM25, and hybrid (RRF) retrieval.

Components:
- Semantic retriever (FAISS + sentence-transformers via LangChain)
- BM25 retriever (wrapped around the bm25.py implementation)
- Hybrid retriever using Reciprocal Rank Fusion (RRF)
- Context builder
- Prompt template
- LLM (HuggingFaceEndpoint wrapped in ChatHuggingFace)
- LCEL RAG chains (semantic + hybrid)
"""
from typing import List, Dict, Any, Optional
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_community.vectorstores import FAISS

from src.bm25 import build_bm25, load_bm25, bm25_search
from src.semantic import (
    build_embeddings,
    build_faiss_index,
    save_faiss,
    load_faiss,
    semantic_search,
)


# 2.3 Prompt template
SYSTEM_PROMPT = """
You are a helpful Amazon shopping assistant.
Answer the question using ONLY the following context (real product reviews + metadata).
If the answer is not in the context, say you don't know.
Always cite the product ASIN when possible.
Be concise and directly address the question.
""".strip()


def build_prompt_template(system_prompt: str = SYSTEM_PROMPT) -> ChatPromptTemplate:
    """
    Build a ChatPromptTemplate for the RAG pipeline.

    Variables:
        - context: injected from retrieved documents
        - question: the user query
    """
    template = """{system_prompt}

context:
{context}

question:
{question}

Answer based ONLY on the context above:"""

    return ChatPromptTemplate.from_template(template).partial(
        system_prompt=system_prompt
    )


# 2.2 Context building
def build_context(docs: List[Document]) -> str:
    """
    Convert retrieved documents into a clean, structured context block.

    Assumes each Document has:
        - page_content: text (metadata or review)
        - metadata: may contain fields like:
            - source: "metadata" or "review"
            - asin
            - rating
            - verified_purchase
            - main_category
            - categories
    """
    blocks = []
    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("source", "unknown")
        asin = meta.get("asin", "N/A")
        rating = meta.get("rating", None)
        rating_str = f"{rating}/5" if rating is not None else "N/A"
        verified = meta.get("verified_purchase", None)
        verified_str = "Yes" if verified else ("No" if verified is not None else "N/A")
        main_category = meta.get("main_category", "")
        categories = meta.get("categories", "")

        header_lines = [
            f"Source: {source}",
            f"ASIN: {asin}",
        ]

        if main_category:
            header_lines.append(f"Main Category: {main_category}")
        if categories:
            header_lines.append(f"Categories: {categories}")
        if source == "review":
            header_lines.append(f"Rating: {rating_str}")
            header_lines.append(f"Verified Purchase: {verified_str}")

        block = (
            "\n".join(header_lines)
            + "\n\nText:\n"
            + doc.page_content
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


# 2.1 Semantic retriever (LangChain)
def build_semantic_retriever(
    docs: List[Document],
    k: int = 5,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Runnable:
    """
    Build a semantic search retriever using FAISS + sentence-transformers via LangChain.

    Args:
        docs: List of LangChain Document objects.
        k: Top-k documents to retrieve.
        model_name: HuggingFace sentence-transformers model.

    Returns:
        A LangChain retriever (Runnable) that maps str -> List[Document].
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever


# 3.1 BM25 retriever wrapper
class BM25RetrieverLC(Runnable):
    """
    LangChain-style retriever wrapper around the custom BM25 implementation in bm25.py.

    Expects:
        - a BM25 index
        - the original documents list
        - top-k parameter
    """

    def __init__(self, bm25, documents: List[Document], k: int = 5):
        self.bm25 = bm25
        self.documents = documents
        self.k = k

    def invoke(self, query: str, config: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Run BM25 search and return top-k Documents.
        """
        results = bm25_search(query, self.bm25, self.documents, k=self.k)
        # bm25_search returns list of dicts with "text" and "metadata"
        docs = [
            Document(page_content=r["text"], metadata=r["metadata"])
            for r in results
        ]
        return docs


def build_bm25_retriever(
    docs: List[Document],
    k: int = 5,
) -> BM25RetrieverLC:
    """
    Build a BM25 retriever using the bm25.py utilities.

    Args:
        docs: List of LangChain Document objects.
        k: Top-k documents to retrieve.

    Returns:
        BM25RetrieverLC instance.
    """
    bm25, tokenized = build_bm25(docs)
    # save_bm25(bm25, tokenized, save_dir="../bm25_index/")
    return BM25RetrieverLC(bm25=bm25, documents=docs, k=k)


# 3.3 Hybrid retriever with RRF
class HybridRetriever(Runnable):
    """
    Hybrid retriever that combines BM25 and semantic retrievers using Reciprocal Rank Fusion (RRF).

    Both underlying retrievers must return List[Document] on .invoke(query).
    """

    def __init__(
        self,
        bm25_retriever: Runnable,
        semantic_retriever: Runnable,
        k: int = 5,
        rrf_k: int = 60,
    ):
        """
        Args:
            bm25_retriever: Runnable returning List[Document].
            semantic_retriever: Runnable returning List[Document].
            k: Final top-k documents to return.
            rrf_k: Constant used in RRF scoring (larger smooths rank differences).
        """
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.k = k
        self.rrf_k = rrf_k

    def _doc_key(self, doc: Document) -> Any:
        """
        Key used to identify duplicates across retrievers.

        Prefer ASIN if present, otherwise fall back to (page_content, source).
        """
        meta = doc.metadata or {}
        asin = meta.get("asin")
        if asin is not None:
            return ("asin", asin)
        source = meta.get("source", "")
        return ("content", doc.page_content, source)

    def invoke(self, query: str, config: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Run both retrievers, fuse results with RRF, and return top-k Documents.
        """
        bm25_docs = self.bm25_retriever.invoke(query)
        semantic_docs = self.semantic_retriever.invoke(query)

        rankings: Dict[Any, Dict[str, int]] = defaultdict(dict)
        doc_store: Dict[Any, Document] = {}

        # Record ranks from BM25
        for rank, doc in enumerate(bm25_docs, start=1):
            key = self._doc_key(doc)
            doc_store[key] = doc
            rankings[key]["bm25"] = rank

        # Record ranks from semantic
        for rank, doc in enumerate(semantic_docs, start=1):
            key = self._doc_key(doc)
            # Prefer first occurrence of the document if already seen
            if key not in doc_store:
                doc_store[key] = doc
            rankings[key]["semantic"] = rank

        # Compute RRF scores
        scores: Dict[Any, float] = {}
        for key, rank_dict in rankings.items():
            score = 0.0
            for r in rank_dict.values():
                score += 1.0 / (self.rrf_k + r)
            scores[key] = score

        # Sort by RRF score descending
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

        # Take top-k documents
        top_docs = [doc_store[key] for key in sorted_keys[: self.k]]
        return top_docs


def build_hybrid_retriever(
    docs: List[Document],
    k: int = 5,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    rrf_k: int = 60,
) -> HybridRetriever:
    """
    Build a hybrid retriever that combines BM25 and semantic retrievers using RRF.

    Args:
        docs: List of LangChain Document objects.
        k: Top-k documents to return.
        model_name: Sentence-transformers model for semantic retriever.
        rrf_k: RRF constant.

    Returns:
        HybridRetriever instance.
    """
    bm25_ret = build_bm25_retriever(docs, k=k)
    semantic_ret = build_semantic_retriever(docs, k=k, model_name=model_name)
    hybrid = HybridRetriever(
        bm25_retriever=bm25_ret,
        semantic_retriever=semantic_ret,
        k=k,
        rrf_k=rrf_k,
    )
    return hybrid


# LLM construction
def build_llm(
    repo_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    max_new_tokens: int = 50,
    task: str = "text-generation",
    provider: str = "auto",
) -> ChatHuggingFace:
    """
    Build the ChatHuggingFace LLM wrapper around HuggingFaceEndpoint.

    Please set HUGGINGFACEHUB_API_TOKEN in the environment.
    """
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        task=task,
        max_new_tokens=max_new_tokens,
        provider=provider,
    )
    llm = ChatHuggingFace(llm=llm_endpoint)
    return llm


# 2.4 / 3.4 RAG pipelines
def build_semantic_rag_chain(
    docs: List[Document],
    k: int = 5,
    system_prompt: str = SYSTEM_PROMPT,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Runnable:
    """
    Build a RAG chain using only the semantic retriever.

    Pipeline:
        query (str)
          -> semantic retriever
          -> build_context
          -> prompt template
          -> LLM
          -> StrOutputParser
    """
    retriever = build_semantic_retriever(docs, k=k, model_name=model_name)
    prompt_template = build_prompt_template(system_prompt)
    llm = build_llm()

    rag_chain: Runnable = (
        {
            "context": retriever | build_context,
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain


def build_hybrid_rag_chain(
    docs: List[Document],
    k: int = 5,
    system_prompt: str = SYSTEM_PROMPT,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    rrf_k: int = 60,
) -> Runnable:
    """
    Build a RAG chain using the hybrid retriever (BM25 + semantic with RRF).

    Pipeline:
        query (str)
          -> hybrid retriever
          -> build_context
          -> prompt template
          -> LLM
          -> StrOutputParser
    """
    hybrid_retriever = build_hybrid_retriever(
        docs, k=k, model_name=model_name, rrf_k=rrf_k
    )
    prompt_template = build_prompt_template(system_prompt)
    llm = build_llm()

    rag_chain: Runnable = (
        {
            "context": hybrid_retriever | build_context,
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    print(
        "RAG pipeline module. Import and use "
        "build_semantic_rag_chain or build_hybrid_rag_chain in your notebook or scripts"
    )
