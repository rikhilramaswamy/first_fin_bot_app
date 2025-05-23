from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage, HumanMessage

import bs4
from langchain_community.document_loaders import RecursiveUrlLoader

from urllib.request import Request, urlopen
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import ssl
import os
import sys
import streamlit as st
from constants import GEMINI_API_KEY, LLM_MODEL_NAME, SITEMAP_URL
import pysqlite3
import sys
import time

# chromaDB requires sqlite3 on streamlit platform
# this fixes sqlite3 library install/dependency issue
sys.modules["sqlite3"] = pysqlite3
import chromadb

def get_sitemap(url):
    req = Request(
        url=url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    response = urlopen(req)
    xml = BeautifulSoup(
        response,
        "lxml-xml",
        from_encoding=response.info().get_param("charset")
    )
    return xml

def get_urls(xml, name=None, data=None, verbose=False):
    urls = []
    for url in xml.find_all("url"):
        if xml.find("loc"):
            loc = url.findNext("loc").text
            urls.append(loc)
        if len(urls) > 5:
            break
    return urls

def scrape_site(url = "https://zerodha.com/varsity/chapter-sitemap2.xml"):
    ssl._create_default_https_context = ssl._create_stdlib_context
    xml = get_sitemap(url)
    urls = get_urls(xml, verbose=False)

    docs = []
    print("scarping the website ...")
    for i, url in enumerate(urls):
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    return docs

def embed_with_retry(embedding_fn, texts, max_retries=3):
    for attempt in range(max_retries):
        try:
            return embedding_fn(texts)
        except Exception as e:
            st.warning(f"Embedding batch failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    raise Exception("Embedding failed after retries.")

@st.cache_resource # Cache the creation of vector store if documents are processed in-app
def vector_retriever(_docs):
    st.write("--- Inside vector_retriever function ---")

    # 1. Reduce chunk size for faster embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(_docs)
    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="RETRIEVAL_DOCUMENT",
        embed_kwargs={"output_dimensionality": 512}
    )

    persistent_db_path = os.path.join(os.getcwd(), "mydb.chromadb")
    BATCH_SIZE = 2  # 2. Lower batch size to avoid timeouts

    all_embeddings = []
    all_metadatas = []
    all_texts = []

    for i in range(0, len(splits), BATCH_SIZE):
        batch = splits[i:i+BATCH_SIZE]
        batch_texts = [doc.page_content for doc in batch]
        batch_metadatas = [doc.metadata for doc in batch]
        try:
            # 3. Add retry logic for embedding
            batch_embeddings = embed_with_retry(gemini_embeddings.embed_documents, batch_texts)
            all_embeddings.extend(batch_embeddings)
            all_metadatas.extend(batch_metadatas)
            all_texts.extend(batch_texts)
        except Exception as e:
            st.warning(f"Batch {i//BATCH_SIZE+1} failed after retries: {e}")

    vectorstore = Chroma(
        embedding_function=gemini_embeddings,
        persist_directory=persistent_db_path
    )
    vectorstore.add_embeddings(
        embeddings=all_embeddings,
        metadatas=all_metadatas,
        documents=all_texts
    )
    vectorstore.persist()

    st.write("--- Vector store created/loaded ---")
    return vectorstore.as_retriever()

@st.cache_resource # Cache the entire RAG chain for a given URL
def create_rag_chain(url):
    docs = scrape_site(url)
    retriever = vector_retriever(docs)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME) 

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are a financial assistant for question-answering tasks related to finance or related topics only "
        "Do not answer questions related to any other topics except finance"
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "If the question is not clear ask follow up questions"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain
      

st.title("RAG based Financial ChatBot")

# Set environment variables
os.environ['GEMINI_API_KEY'] = st.secrets["GEMINI_API_KEY"]

# store the rag_chain object INSTEAD of fetching data and/or creating rag_chain object
# on every LLM request 
# IOW: create_chain() API is invoked only on APP init for the first time
# on subsequent query rag_chain object created on init is re-used
if 'rag_chain' not in st.session_state:
    st.session_state['rag_chain'] = create_rag_chain(SITEMAP_URL)

# use session state to store chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if user_input := st.chat_input("Please ask your question!:"):
    response = st.session_state['rag_chain'].invoke({"input": user_input,
                                                     "chat_history": st.session_state['messages']}) 
    # Append the user input and bot response to the messages list
    st.session_state['messages'].extend(
        [HumanMessage(user_input), 
         AIMessage(response["answer"])])

    st.write(response["answer"])
