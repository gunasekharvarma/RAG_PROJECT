from typing import List
import boto3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.chat_models import BedrockChat
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document

from embed.AmazonTitan.amazon_embed import TitanTextImageEmbeddings, TitanTextEmbeddings
from generator.base import GeneratorBase
from retriever.re_opensearch import OpenSearchRetriever

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

class BedrockGenerator(GeneratorBase):
    """
    A generator class using AWS Bedrock for output generation, extending GeneratorBase.
    Supports models like Anthropic Claude or Meta Llama via BedrockChat.
    """

    def __init__(
        self,
        aws_access_key_id: str = AWS_ACCESS_KEY_ID,
        aws_secret_access_key: str = AWS_SECRET_ACCESS_KEY,
        region_name: str = "us-east-1",
        # model_id: str = "anthropic.claude-v2",  # Example: Claude v2; adjust as needed
        model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
    ):
        """
        Initialize the BedrockGenerator.

        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param region_name: AWS region (default: "us-east-1").
        :param model_id: Bedrock model ID (default: "anthropic.claude-v2").
        """
        # self.llm = BedrockChat(
        #     model_id=model_id,
        #     region_name=region_name,
        #     credentials_profile_name=None,  # Use explicit keys
        #     aws_access_key_id=aws_access_key_id,
        #     aws_secret_access_key=aws_secret_access_key,
        # )
        boto3_session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=region_name,
        )

        self.llm = BedrockChat(
            model_id=model_id,
            client=boto3_session.client("bedrock-runtime")
        )

    def generate_output(self, docs: List[Document], question: str) -> str:
        """
        Generate output using Bedrock based on retrieved documents and the question.

        :param docs: List of retrieved Document objects.
        :param question: The user's question.
        :return: The generated answer string.
        """
        prompt = self._frame_prompt(docs, question)
        response = self.llm.invoke(prompt)
        return response.content.strip()

if __name__ == '__main__' :
    query = "What is Self Attention"
    embeddings = TitanTextImageEmbeddings()#model_id="amazon.titan-embed-text-v1")
    print("Embeddings are working for query")
    # run_manager = CallbackManagerForRetrieverRun()
    vectorstore_for_OSVS = OpenSearchVectorSearch(
        opensearch_url="https://search-guna-domain-dokyig3bsq3pjgjynjaggzoaqa.aos.us-east-1.on.aws",
        index_name='rag-test',
        embedding_function=embeddings,
        http_auth=("admin", "Admin123$")
    )
    os_retriever = OpenSearchRetriever(
        vectorstore=vectorstore_for_OSVS,
        top_k=4
    )
    print("Documents are Fetched Successfully")
    bedrock = BedrockGenerator(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
    )
    print("BedRock Asssignment")
    retrieved_docs = os_retriever.get_relevant_documents(query=query)
    print("Getting relevant Documents")
    answer = bedrock.generate_output(retrieved_docs,query)
    print("Answer Generated Successfully")
    print(answer)