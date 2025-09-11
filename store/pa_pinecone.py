import logging
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from embed.AmazonTitan.amazon_embed import TitanTextImageEmbeddings
from extract.LlamaParse.extract_pages import LlamaParseLoader
from store.base import StoreBase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

class PineconeStore(StoreBase):
    """
    A class for storing embedded documents into Pinecone, extending StoreBase.
    """

    def store(
        self,
        index_name: str,
        pinecone_api_key: str = PINECONE_API_KEY,
        namespace: str = "",
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> PineconeClient.Index:
        """
        Store the embedded documents into Pinecone.

        :param pinecone_api_key: API key for Pinecone.
        :param index_name: Name of the Pinecone index to use or create.
        :param namespace: Optional namespace in Pinecone index (default: "").
        :param cloud: Cloud provider for Pinecone serverless index (default: "aws").
        :param region: Region for Pinecone serverless index (default: "us-east-1").
        :return: The Pinecone index instance.
        """
        logger.info(f"Starting store operation for index: {index_name}, namespace: {namespace}")

        # Initialize Pinecone client
        logger.debug("Initializing Pinecone client")
        try:
            pc = PineconeClient(api_key=pinecone_api_key)
            logger.debug("Pinecone client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            raise

        # Check if index exists; create if not (serverless spec)
        logger.debug("Checking if index exists")
        index_list = pc.list_indexes().names()
        logger.debug(f"Existing indexes: {index_list}")
        if index_name not in index_list:
            logger.info(f"Index {index_name} not found, creating new index with dimension: {self.embeddings.dimension}")
            try:
                pc.create_index(
                    name=index_name,
                    dimension=self.embeddings.dimension,  # Use dimension from the embeddings class
                    metric="cosine",
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )
                logger.info(f"Index {index_name} created successfully")
            except Exception as e:
                logger.error(f"Failed to create index {index_name}: {e}")
                raise

        # Get the Pinecone index
        logger.debug(f"Retrieving index: {index_name}")
        try:
            index = pc.Index(name=index_name)
            logger.debug(f"Index {index_name} retrieved successfully")
        except Exception as e:
            logger.error(f"Failed to retrieve index {index_name}: {e}")
            raise

        # Prepare and upsert text vectors
        logger.debug("Preparing text upserts")
        text_upserts = self._prepare_text_upserts()
        if text_upserts:
            logger.info(f"Upserting {len(text_upserts)} text vectors to namespace: {namespace}")
            try:
                index.upsert(vectors=text_upserts, namespace=namespace)
                logger.debug(f"Successfully upserted text vectors")
            except Exception as e:
                logger.error(f"Failed to upsert text vectors: {e}")
                raise
        else:
            logger.debug("No text vectors to upsert")

        # Prepare and upsert image vectors
        logger.debug("Preparing image upserts")
        image_upserts = self._prepare_image_upserts()
        if image_upserts:
            logger.info(f"Upserting {len(image_upserts)} image vectors to namespace: {namespace}")
            try:
                index.upsert(vectors=image_upserts, namespace=namespace)
                logger.debug(f"Successfully upserted image vectors")
            except Exception as e:
                logger.error(f"Failed to upsert image vectors: {e}")
                raise
        else:
            logger.debug("No image vectors to upsert")

        logger.info(f"Store operation completed for index: {index_name}")
        return index


if __name__ == '__main__':
    loader = LlamaParseLoader(
        pdf_path="D:/Learn-RAG-code-only/pdf_files/transformer.pdf",
        describe_images=True,
        image_dir="C:/Users/Sivakumar Keertipati/Desktop/RAG_PROJECT/extract_images"
    )

    titan_embeddings = TitanTextImageEmbeddings()

    pinecone_store = PineconeStore(loader,embeddings=titan_embeddings)
    pinecone_store.store(index_name='rag-test')

