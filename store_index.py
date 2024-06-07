from src.helper import download_hf_embeddings, text_split, load_pdf
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OpenAI_API_KEY = os.environ.get('OpenAI_API_KEY')

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hf_embeddings()

index_name = "dharmallama"

texts = [t.page_content for t in text_chunks]
docsearch = PineconeVectorStore.from_texts(
    texts,
    index_name=index_name,
    embedding=embeddings
)
