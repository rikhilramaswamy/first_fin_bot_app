from langchain_google_genai import ChatGoogleGenerativeAI
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

from sentence_transformers import SentenceTransformer  # For local embeddings
import numpy as np

# chromaDB requires sqlite3 on streamlit platform
sys.modules["sqlite3"] = pysqlite3
import chromadb

LAST_N_CHATS = 10

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
        if len(urls) > 1:
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

def embed_with_retry(embedding_fn, texts, max_retries=5):
    for attempt in range(max_retries):
        try:
            return embedding_fn(texts)
        except Exception as e:
            st.warning(f"Embedding batch failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    raise Exception("Embedding failed after retries.")

@st.cache_resource
def vector_retriever(_docs):
    st.write("--- Inside vector_retriever function ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=64)
    splits = text_splitter.split_documents(_docs)
    doc_texts = [doc.page_content for doc in splits]
    doc_metadatas = [doc.metadata for doc in splits]

    # Use SentenceTransformer for local embeddings
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    BATCH_SIZE = 8

    # No need to precompute embeddings; let Chroma handle it via the embedding function
    persistent_db_path = os.path.join(os.getcwd(), "mydb.chromadb")
    vectorstore = Chroma.from_documents(
        documents=doc_texts,
        embedding=lambda texts: embedder.encode(texts).tolist(),
        metadatas=doc_metadatas,
        persist_directory=persistent_db_path
    )

    st.write("--- Vector store created/loaded ---")

    # Custom retriever that uses the same embedder for query embedding
    class SentenceTransformerRetriever:
        def __init__(self, vectorstore, embedder):
            self.vectorstore = vectorstore
            self.embedder = embedder

        def get_relevant_documents(self, query):
            query_emb = self.embedder.encode([query])[0]
            results = self.vectorstore.similarity_search_by_vector(query_emb)
            return results

        # For LangChain compatibility
        def __call__(self, query):
            return self.get_relevant_documents(query)

    return SentenceTransformerRetriever(vectorstore, embedder)

@st.cache_resource
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

os.environ['GEMINI_API_KEY'] = st.secrets["GEMINI_API_KEY"]

if 'rag_chain' not in st.session_state:
    st.session_state['rag_chain'] = create_rag_chain(SITEMAP_URL)

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if user_input := st.chat_input("Please ask your question!:"):
    response = st.session_state['rag_chain'].invoke({
        "input": user_input,
        "chat_history": st.session_state['messages']
    })
    st.session_state['messages'].extend(
        [HumanMessage(user_input), AIMessage(response["answer"])]
    )
    st.session_state['messages'] = st.session_state['messages'][-LAST_N_CHATS:]
    st.write(response["answer"])
