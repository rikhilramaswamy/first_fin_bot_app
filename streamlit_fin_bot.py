from langchain_openai import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_community.document_loaders import RecursiveUrlLoader

from urllib.request import Request, urlopen
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import ssl
import os
import sys
import streamlit as st
from constants import OPENAI_API_KEY, LLM_MODEL_NAME


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
	vectorstore = Chroma.from_documents(documents=splits,
	                                    embedding=OpenAIEmbeddings())
	return vectorstore.as_retriever()

def create_chain():
	docs = scrape_site()
	retriever = vector_retriever(docs)
	# 2. Incorporate the retriever into a question-answering chain.
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

	prompt = ChatPromptTemplate.from_messages(
	    [
	        ("system", system_prompt),
	        ("human", "{input}"),
	    ]
	)

	llm = ChatOpenAI(model=LLM_MODEL_NAME)

	question_answer_chain = create_stuff_documents_chain(llm, prompt)
	return create_retrieval_chain(retriever, question_answer_chain)


st.title("RAG based Financial ChatBot")

# Set environment variables
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


# Add a text input widget to get input from the user
user_input = st.text_input("Enter some text:", "Hello, Streamlit!")

rag_chain = create_chain()

response = rag_chain.invoke({"input": user_input})

# Add a button. The st.button() function returns True if the button was clicked,
# and False otherwise.
if st.button("OK_Input"):
    # This block of code will only execute when the "OK_Input" button is clicked.
    st.write("LLM response is:")
    st.write(response["answer"])


'''
if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Expected two arguments OPENAI_API_KEY and the question")
		exit(1)

	print(len(sys.argv))

	os.environ["OPENAI_API_KEY"] = sys.argv[1]
	rag_chain = create_chain()
	response = rag_chain.invoke({"input": sys.argv[2]})
	print("-----------------")
	print("Answer:")
	print(response["answer"])
'''

