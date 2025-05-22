from langchain_openai import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
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
from constants import OPENAI_API_KEY, LLM_MODEL_NAME, SITEMAP_URL
import pysqlite3
import sys

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
		if i == 10:
			break
	return docs


def vector_retriever(docs):
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
												   chunk_overlap=200)
	splits = text_splitter.split_documents(docs)
	oi_embeddings = OpenAIEmbeddings()
	vectorstore = Chroma.from_documents(documents=splits,
										embedding=oi_embeddings)
	return vectorstore.as_retriever()

def create_rag_chain(url):
	docs = scrape_site(url)
	retriever = vector_retriever(docs)
	# 2. Incorporate the retriever into a question-answering chain.

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

	llm = ChatOpenAI(model=LLM_MODEL_NAME)

	history_aware_retriever = create_history_aware_retriever(
		llm, retriever, contextualize_q_prompt
	)

	# Example of how to create a QA prompt (ensure system_prompt is defined)
	system_prompt = (
		"You are a financial assistant for question-answering tasks. "
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
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]


# Add a text input widget to get input from the user
user_input = st.text_input("Enter your question below:", "Ask, Bot!")

# store the rag_chain object INSTEAD of fetching data and/or creating rag_chain object
# on every LLM request 
# IOW: create_chain() API is invoked only on APP init for the first time
# on subsequent query rag_chain object created on init is re-used
if 'rag_chain' not in st.session_state:
	st.session_state['rag_chain'] = create_rag_chain(SITEMAP_URL)
	st.write("invoking data fetch to create rag chain")

# use session state to store chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []


response = st.session_state['rag_chain'].invoke({"input": user_input,
                                                 "chat_history": st.session_state['messages']}) 
# Append the user input and bot response to the messages list
st.session_state['messages'].extend(
    [HumanMessage(user_input), 
    AIMessage(response["answer"])])

# Add a button. The st.button() function returns True if the button was clicked,
# and False otherwise.
if st.button("Ask Question"):
    # This block of code will only execute when the "OK_Input" button is clicked.
    st.write(response["answer"])
