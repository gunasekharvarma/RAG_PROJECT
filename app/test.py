from typing import List

import boto3
from fastapi.exceptions import HTTPException
from fastapi import FastAPI
from langchain_community.chat_models import BedrockChat
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel

from embed.AmazonTitan.amazon_embed import TitanTextImageEmbeddings
from generator.ge_aws_bedrock import BedrockGenerator
from retriever.re_opensearch import OpenSearchRetriever
from retriever.re_rerank import RerankedRetriever

app = FastAPI(title = "Retrieve")

class QueryRequest(BaseModel):
    query:str
    use_detailed_rag: bool = False

class QueryResponse(BaseModel):
    context: List[dict]
    llm_output: str
    metadata: dict = None

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
COHERE_API_KEY = ""
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""

reranked_retriever = RerankedRetriever(os_retriever,cohere_api_key=COHERE_API_KEY)

def format_docs(docs):
    """Format retrieved documents into a single context string."""
    return "\n\n".join([
        f"Document {i + 1}:\n{doc.page_content}\n"
        f"Metadata: {doc.metadata}"
        for i, doc in enumerate(docs)
    ])

rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant that answers questions based on the provided context. 
Use the following pieces of context to answer the question. If you don't know the answer 
based on the context, just say that you don't know.

Context:
{context}

Question: {question}

Answer: Provide a comprehensive and accurate answer based on the context above.
""")

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",  # change if needed
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

bedrock = BedrockGenerator(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
)

llm = BedrockChat(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    client=bedrock_client,
)

rag_chain = (

    {
        'question' : RunnablePassthrough(),
        'context' : reranked_retriever  | format_docs
    }
    | rag_prompt
    | llm
    | StrOutputParser()

)

class SimpleResponseModel(BaseModel):
    answer: str


@app.post('/simple-rag',response_model=SimpleResponseModel)
def simple_rag(request: QueryRequest):

    try:
        answer = rag_chain.invoke(request.query)
        return {'answer' : answer}
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))
