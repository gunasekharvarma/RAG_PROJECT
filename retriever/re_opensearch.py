import os
import sys

from openai import vector_stores

sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'),'..')))
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.vectorstores.opensearch_vector_search import OpenSearchVectorSearch

from embed.AmazonTitan.amazon_embed import TitanTextImageEmbeddings


class OpenSearchRetriever(BaseRetriever):
    """
    A retriever class that extends BaseRetriever for querying an OpenSearch vectorstore.
    This class uses the provided embeddings to embed queries and retrieve relevant documents.
    """
    vectorstore: OpenSearchVectorSearch
    top_k: int = 4
    min_score: float = 0.0

    # def __init__(
    #     self,
    #     vectorstore: OpenSearchVectorSearch,
    #     top_k: int = 4,
    #     min_score: float = 0.0,
    # ):
    """
    Initialize the OpenSearchRetriever.

    :param vectorstore: The LangChain OpenSearchVectorSearch instance.
    :param top_k: Number of top results to return (default: 4).
    :param min_score: Minimum similarity score threshold (default: 0.0).
    """
        #super().__init__()
        # self.vectorstore = vectorstore
        # self.top_k = top_k
        # self.min_score = min_score



    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """
        Retrieve relevant documents for a given query.

        :param query: The search query.
        :param run_manager: Callback manager for the retriever run.
        :return: List of retrieved Document objects.
        """
        # Perform similarity search with score
        results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
        # Filter by min_score (note: scores in OpenSearch are typically cosine similarity, range -1 to 1)
        docs = [doc for doc, score in results if score >= self.min_score]
        return docs

if __name__ == '__main__' :

    query = "What is Self Attention"
    embeddings = TitanTextImageEmbeddings()
    vectorstore_for_OSVS = OpenSearchVectorSearch(
            opensearch_url="https://search-guna-domain-dokyig3bsq3pjgjynjaggzoaqa.aos.us-east-1.on.aws",
            index_name='rag-test',
            embedding_function=embeddings,
            http_auth=("admin", "Admin123$")
        )
    os_retriever = OpenSearchRetriever(
        vectorstore = vectorstore_for_OSVS,
        top_k=4
    )

    retrieved_docs = os_retriever.get_relevant_documents(query=query)
    print(retrieved_docs)

