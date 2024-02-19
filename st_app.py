import requests
from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer
import time
import wikipedia
import streamlit as st
import chromadb
from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

from bs4 import BeautifulSoup


### Initializing Model and Embeddings
model_id = 'llama-2-13b-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        torch_dtype=torch.float16,
        device_map="auto",)

llm = HuggingFacePipeline(pipeline=query_pipeline)
# checking again that everything is working fine


##### Embedding Building #####
def get_wiki():
    # set language to English (default is auto-detect)
    lang = "en"

    """
    fetching summary from wikipedia
    """
    # set language to English (default is auto-detect)
    #summary = wikipedia.summary(search, sentences = 5)

    """
    scrape wikipedia page of the requested query
    """

    # create URL based on user input and language
    url = 'https://en.wikipedia.org/wiki/Rio_Tinto_(corporation)'

    # send GET request to URL and parse HTML content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # extract main content of page
    content_div = soup.find(id="mw-content-text")

    # extract all paragraphs of content
    paras = content_div.find_all('p')

    # concatenate paragraphs into full page content
    full_page_content = ""
    for para in paras:
        full_page_content += para.text

    # print the full page content
    return full_page_content

content= get_wiki()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.create_documents(content)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

#### Embedding Store ####
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

def test_rag(qa, query):
    result = qa.run(query)
    return result

    ################################### Start Chat Session  ###################################
def run_st_app():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    if prompt := st.chat_input("Hello?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            m = st.session_state.messages[-1]
            response = test_rag(qa, m['content'])
            full_response = ''
            # Simulate stream of response with milliseconds delay
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            #message_placeholder.markdown(response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    ### Argument parser

    run_st_app()
