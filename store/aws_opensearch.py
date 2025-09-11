import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from embed.AmazonTitan.amazon_embed import TitanTextImageEmbeddings
from extract.LlamaParse.extract_pages import LlamaParseLoader
from store.base import StoreBase
from langchain.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from opensearchpy import OpenSearch

# Configure logging with timestamp, log level, and message format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenSearchStore(StoreBase):
    """
    A class for storing embedded documents into OpenSearch, extending StoreBase.
    Uses LangChain's OpenSearchVectorSearch for integration.
    """

    def store(
        self,
        opensearch_url: str,
        index_name: str,
        http_auth: tuple = None,  # (username, password)
        use_ssl: bool = True,
        verify_certs: bool = True,
    ) -> OpenSearchVectorSearch:
        """
        Store the embedded documents into OpenSearch.

        :param opensearch_url: URL for the OpenSearch instance (e.g., "https://localhost:9200").
        :param index_name: Name of the OpenSearch index to use or create.
        :param http_auth: Optional HTTP authentication tuple (username, password).
        :param use_ssl: Use SSL for connection (default: True).
        :param verify_certs: Verify SSL certificates (default: True).
        :return: The LangChain OpenSearchVectorSearch instance.
        """
        # Log the start of the store operation
        logger.info(f"Starting store operation for OpenSearch index: {index_name}, URL: {opensearch_url}")

        # Initialize OpenSearch client
        logger.debug("Initializing OpenSearch client")
        try:
            client = OpenSearch(
                hosts=[opensearch_url],
                http_auth=http_auth,
                use_ssl=use_ssl,
                verify_certs=verify_certs,
            )
            logger.debug("OpenSearch client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenSearch client: {e}")
            raise

        # Create vectorstore for text documents using LangChain's OpenSearchVectorSearch
        logger.debug(f"Creating vectorstore for {len(self.docs)} text documents")
        try:
            vectorstore = OpenSearchVectorSearch.from_documents(
                documents=self.docs,
                embedding=self.embeddings,  # Use embed_query as the embedding function
                opensearch_url=opensearch_url,
                index_name=index_name,
                http_auth=http_auth,
                use_ssl=use_ssl,
                verify_certs=verify_certs,
            )
            logger.info(f"Vectorstore created for index: {index_name}")
        except Exception as e:
            logger.error(f"Failed to create vectorstore for index {index_name}: {e}")
            raise

        # Prepare and upsert image vectors if available
        logger.debug("Preparing image upserts")
        image_upserts = self._prepare_image_upserts()
        if image_upserts:
            logger.info(f"Upserting {len(image_upserts)} image vectors to index: {index_name}")
            for i, upsert in enumerate(image_upserts):
                doc_id = upsert['id']
                vector = upsert['values']
                metadata = upsert['metadata']
                logger.debug(f"Upserting image {i+1}/{len(image_upserts)} with ID: {doc_id}")
                try:
                    # Upsert image vector to OpenSearch
                    client.index(
                        index=index_name,
                        id=doc_id,
                        body={
                            "vector_field": vector,  # Assuming 'vector_field' is the knn_vector field
                            **metadata
                        }
                    )
                    logger.debug(f"Successfully upserted image vector with ID: {doc_id}")
                except Exception as e:
                    logger.error(f"Failed to upsert image vector with ID {doc_id}: {e}")
                    raise
        else:
            logger.debug("No image vectors to upsert")

        # Log completion of the store operation
        logger.info(f"Store operation completed for index: {index_name}")
        return vectorstore



if __name__ == "__main__":
    loader = LlamaParseLoader(
        pdf_path="D:/Learn-RAG-code-only/pdf_files/transformer.pdf",
        describe_images=True,
        image_dir="C:/Users/Sivakumar Keertipati/Desktop/RAG_PROJECT/extract_images"
    )

    titan_embeddings = TitanTextImageEmbeddings()

    vectorstore = OpenSearchStore(loader, titan_embeddings)
    vectorstore.store(
        opensearch_url="https://search-guna-sekhar-jdvuj4ov6ku5qwxizcrdtz3vhy.aos.us-east-1.on.aws",
        index_name="rag-test",
        http_auth=("admin", "Admin123$"),  # if fine-grained access control enabled
        use_ssl=True,
        verify_certs=True
    )

    # Testing whether the document is stored or not
    # Check if a specific index exists
    if vectorstore.indices.exists("rag-test"):
        print("Index exists ✅")
    else:
        print("Index Not Found ❌")

    count = vectorstore.count(index="rag-test")
    print("Document count:", count["count"])

    # Fetch first 5 docs
    response = vectorstore.search(index="rag-test", body={"size": 5, "query": {"match_all": {}}})
    for hit in response["hits"]["hits"]:
        print(hit["_id"],hit["_source"])



