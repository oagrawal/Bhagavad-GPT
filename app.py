from flask import Flask, render_template, jsonify, request
from src.helper import download_hf_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  
from langchain.chains import RetrievalQA  
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OpenAI_API_KEY = os.environ.get('OpenAI_API_KEY')

embeddings = download_hf_embeddings()

index_name='dharmallama'
docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm = ChatOpenAI(  
    openai_api_key=OpenAI_API_KEY,  
    model_name='gpt-3.5-turbo',  
    temperature=0.3  
) 

qa = RetrievalQA.from_chain_type(  
    llm=llm,  
    chain_type="stuff",  
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
