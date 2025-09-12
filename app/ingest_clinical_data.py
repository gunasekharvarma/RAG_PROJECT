import tempfile
import logging
import os
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from embed.AmazonTitan.amazon_embed import TitanTextImageEmbeddings
from extract.LlamaParse.extract_pages import LlamaParseLoader
from store.aws_opensearch import OpenSearchStore

app = FastAPI(title="Ingest API")
logger = logging.getLogger(__name__)

class IngestRequest(BaseModel):
    s3_bucket: str
    s3_key: str
    opensearch_url: str
    index_name: str
    opensearch_user: str
    opensearch_pass: str

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

s3_client = boto3.client(
    "s3",
    region_name="us-east-1",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# loader = LlamaParseLoader(
#         pdf_path="D:/Learn-RAG-code-only/pdf_files/transformer.pdf",
#         describe_images=True,
#         image_dir="C:/Users/Sivakumar Keertipati/Desktop/RAG_PROJECT/extract_images"
#     )
#
# titan_embeddings = TitanTextImageEmbeddings()
#
# vectorstore = OpenSearchStore(loader, titan_embeddings)
# vectorstore.store(
#     opensearch_url="https://search-guna-domain-dokyig3bsq3pjgjynjaggzoaqa.aos.us-east-1.on.aws",
#     index_name="rag-test",
#     http_auth=("admin", "Admin123$"),  # if fine-grained access control enabled
#     use_ssl=True,
#     verify_certs=True
# )

def generate_presigned_url(bucket: str, key: str, expiry: int = 3600) -> str:
    """Generate a presigned URL for the S3 object."""
    return s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=expiry
    )

@app.post("/ingest")
async def DataIngestion(request: IngestRequest,):
    try:
        # 1. Download file from S3 → temp path
        # tmp_path = tempfile.mktemp(suffix=".pdf")
        # s3_client.download_file(request.s3_bucket, request.s3_key, tmp_path)
        presigned_url = generate_presigned_url(request.s3_bucket, request.s3_key)
        logger.info(f"Generated presigned URL: {presigned_url}")


        # 2. Load PDF with LlamaParseLoader
        loader = LlamaParseLoader(
            pdf_path=presigned_url,
            describe_images=True,
            image_dir="./extracted_images"  # save parsed images locally if needed
        )

        # 3. Titan Embeddings
        titan_embeddings = TitanTextImageEmbeddings()

        # 4. OpenSearch store
        vectorstore = OpenSearchStore(loader, titan_embeddings)
        vectorstore.store(
            opensearch_url=request.opensearch_url,
            index_name=request.index_name,
            http_auth=(request.opensearch_user, request.opensearch_pass),
            use_ssl=True,
            verify_certs=True
        )

        return {"message": "Document ingested successfully", "s3_key": request.s3_key}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
