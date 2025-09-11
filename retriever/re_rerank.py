from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank  # Updated import
from langchain_core.retrievers import BaseRetriever
from pydantic import SecretStr

"""
The ContextualCompressionRetriever is a component in the LangChain framework, designed to enhance the retrieval process in Retrieval-Augmented Generation (RAG) systems by compressing or filtering retrieved documents to provide more relevant and concise results. It wraps around a base retriever and applies a post-processing step to refine the retrieved documents based on the query context, improving efficiency and relevance in downstream tasks like question answering or text generation.
Key Features and Purpose

Compression and Filtering: The ContextualCompressionRetriever takes documents retrieved by a base retriever (e.g., a vector store retriever like Pinecone or OpenSearch) and processes them using a compressor to extract only the most relevant parts or filter out less relevant documents.
Improved Relevance: It ensures that the retrieved content is more focused on the query, reducing noise and irrelevant information.
Efficiency: By compressing or filtering documents, it reduces the amount of text passed to language models, lowering computational costs and improving response quality.
Modularity: It works with any LangChain retriever and supports various compression techniques, such as LLM-based filtering or embeddings-based re-ranking.

How It Works

Base Retriever: The ContextualCompressionRetriever relies on a base retriever (e.g., PineconeStore or OpenSearchStore from your previous code) to fetch an initial set of documents from a vector store or database based on a query.
Compressor: A compressor (e.g., LLMChainExtractor, DocumentCompressorPipeline, or EmbeddingsFilter) processes the retrieved documents to:

Extract relevant excerpts (e.g., using an LLM to identify key sentences).
Re-rank documents based on relevance (e.g., using embeddings similarity).
Filter out documents below a relevance threshold.


Output: The compressed or filtered documents are returned, typically as a smaller, more relevant set, ready for use in a RAG pipeline.
"""

class RerankedRetriever(ContextualCompressionRetriever):
    """
    A reranked retriever class using CohereRerank for contextual compression.
    This class wraps a base retriever with a Cohere reranker to improve relevance
    by reranking the initial retrieved documents.

    How CohereRerank Works:
    ----------------------
    1. **Initial Retrieval**: The base retriever (e.g., PineconeRetriever) first
       retrieves a larger set of potentially relevant documents using vector similarity.

    2. **Semantic Reranking**: CohereRerank then uses Cohere's reranking models to
       re-score these documents based on their semantic relevance to the query. Unlike
       simple vector similarity, reranking models are specifically trained to understand
       the relationship between queries and passages.

    3. **Cross-Attention Mechanism**: The reranking model uses cross-attention to
       directly compare the query with each document, considering:
       - Lexical overlap and matching
       - Semantic similarity beyond embeddings
       - Query-document interaction patterns
       - Contextual understanding of relevance

    4. **Relevance Scoring**: Each document receives a relevance score from 0 to 1,
       where higher scores indicate better relevance to the query.

    5. **Top-N Selection**: Finally, only the top_n highest-scoring documents are
       returned, ensuring the most relevant results reach the downstream application.

    Benefits:
    ---------
    - **Higher Precision**: Reranking often significantly improves the relevance of
      retrieved documents compared to vector similarity alone
    - **Better Handling of Nuanced Queries**: Understands complex query intents that
      might be missed by embedding-based retrieval
    - **Reduced Hallucination**: By providing more relevant context, it helps reduce
      hallucination in downstream LLM applications

    Models Available:
    ----------------
    - rerank-english-v3.0: Optimized for English text, best performance
    - rerank-multilingual-v3.0: Supports multiple languages
    - rerank-english-v2.0: Older version, still effective
    """

    def __init__(
            self,
            base_retriever: BaseRetriever,
            cohere_api_key: str,
            top_n: int = 3,
            model: str = "rerank-english-v3.0",
    ):
        """
        Initialize the RerankedRetriever.

        :param base_retriever: The base retriever instance (e.g., PineconeRetriever or OpenSearchRetriever).
        :param cohere_api_key: API key for Cohere reranking service.
        :param top_n: Number of top documents to return after reranking (default: 3).
        :param model: Cohere reranking model to use (default: "rerank-english-v3.0").
        """
        compressor = CohereRerank(
            cohere_api_key=SecretStr(cohere_api_key),
            top_n=top_n,
            model=model
        )
        super().__init__(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )