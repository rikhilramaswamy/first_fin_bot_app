from langchain_openai import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Import Document type
from google.api_core import exceptions # For more specific error handling

import bs4
from langchain_community.document_loaders import RecursiveUrlLoader

from urllib.request import Request, urlopen
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import ssl
import os
import sys
import time
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
	return docs


# --- Configuration ---
CHUNK_SIZE = 700  # Smaller chunk size
CHUNK_OVERLAP = 100 # Adjusted overlap
BATCH_SIZE = 30   # Number of document contents to send per embedding API call
MAX_RETRIES = 5
INITIAL_BACKOFF_TIME = 1 # seconds


# --- Helper function for embedding with retry logic ---
@st.cache_data(ttl=600, show_spinner="Generating embeddings...") # Cache for 10 minutes (600 seconds)
def get_embeddings_with_retry(
    embedding_model_instance: GoogleGenerativeAIEmbeddings,
    content_batch: list[str], # Expects a list of strings
    task_type: str,
    output_dimensionality: int,
    max_retries: int = MAX_RETRIES,
    initial_backoff: int = INITIAL_BACKOFF_TIME
) -> list[list[float]]: # Returns a list of embedding vectors
    """
    Attempts to get embeddings for a batch of content with retry logic.
    """
    retries = 0
    backoff_time = initial_backoff
    
    # Use the underlying client directly for finer control over API calls
    # This assumes GoogleGenerativeAIEmbeddings has a 'client' attribute
    # which holds the raw Google AI client.
    api_client = embedding_model_instance.client

    while retries < max_retries:
        try:
            response = api_client.embed_content(
                model=embedding_model_instance.model_name, # Use model name from instance
                content=content_batch,
                task_type=task_type,
                output_dimensionality=output_dimensionality
            )
            # The embeddings are usually in response.embeddings, and each has a .values attribute
            return [e.values for e in response.embeddings]
        except exceptions.ServiceUnavailable as e:
            st.warning(f"Service Unavailable (503) during embedding, retrying in {backoff_time}s... ({e})")
        except exceptions.DeadlineExceeded as e:
            st.warning(f"Deadline Exceeded (504) during embedding, retrying in {backoff_time}s... ({e})")
        except exceptions.ResourceExhausted as e: # Catch 429 specifically for rate limits
            st.warning(f"Rate Limit Exceeded (429) during embedding, retrying in {backoff_time}s... ({e})")
            # Increase backoff more aggressively for rate limits
            backoff_time = min(backoff_time * 3, 60) # Max 60 seconds backoff for rate limits
        except Exception as e: # Catch any other unexpected errors
            st.error(f"An unexpected error occurred during embedding: {e}")
            raise # Re-raise if it's not a recoverable error

        time.sleep(backoff_time)
        backoff_time *= 2 # Exponential backoff for subsequent retries
        retries += 1
    
    raise Exception(f"Failed to get embeddings after {max_retries} retries.")

@st.cache_resource  # Cache the creation of vector store if documents are processed in-app
def vector_retriever(_docs: list[Document]):
    st.write("--- Inside vector_retriever function ---")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(_docs)
    st.write(f"Split {len(_docs)} documents into {len(splits)} chunks.")

    gemini_embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        task_type="RETRIEVAL_DOCUMENT",
        embed_kwargs={"output_dimensionality": 512} # Still pass this for initialization
    )

    persistent_db_path = os.path.join(os.getcwd(), "mydb.chromadb")
    
    # Ensure the directory exists
    os.makedirs(persistent_db_path, exist_ok=True)

    # Initialize Chroma DB (or load if it already exists)
    # If the DB doesn't exist, Chroma will create it.
    # If it exists, we'll add new embeddings to it.
    vectorstore = Chroma(
        persist_directory=persistent_db_path,
        embedding_function=gemini_embeddings_model # Pass the embedding function
    )

    # --- Manual Batching for Embedding and Adding to Chroma ---
    st.write(f"Starting embedding process with batch size: {BATCH_SIZE}...")
    
    # Store documents and their IDs to add to Chroma later
    # Chroma.add_embeddings requires ids, embeddings, and metadatas
    # If you need specific IDs for your chunks, generate them here
    # For simplicity, we'll let Chroma generate them or use sequential numbers
    
    all_chunk_texts = [s.page_content for s in splits]
    all_chunk_metadatas = [s.metadata for s in splits]
    
    # List to store all generated embeddings
    all_generated_embeddings = []
    
    # Iterate through chunks in batches
    for i in range(0, len(all_chunk_texts), BATCH_SIZE):
        chunk_batch_texts = all_chunk_texts[i : i + BATCH_SIZE]
        
        try:
            # Use the retry helper function for embedding
            batch_embeddings = get_embeddings_with_retry(
                embedding_model_instance=gemini_embeddings_model,
                content_batch=chunk_batch_texts,
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=512 # Ensure this matches what you're expecting
            )
            all_generated_embeddings.extend(batch_embeddings)
            st.write(f"Successfully embedded batch {i // BATCH_SIZE + 1} of {len(all_chunk_texts) // BATCH_SIZE + 1}")
        except Exception as e:
            st.error(f"Failed to embed batch {i // BATCH_SIZE + 1} after retries. Error: {e}")
            # Decide if you want to stop or continue with subsequent batches
            # For robust systems, you might log the failed batch and continue
            raise # Re-raise the exception to propagate the error upwards

    # Now add all the generated embeddings and their corresponding documents to Chroma
    # We need to ensure that the embeddings correspond to the original Document objects' content and metadata.
    # Since we processed texts in order, the ordering should match.
    
    if len(all_generated_embeddings) == len(splits):
        st.write(f"Adding {len(all_generated_embeddings)} embeddings to ChromaDB...")
        # Chroma.add_embeddings expects a list of embeddings (list[list[float]])
        # and a list of metadatas (list[dict]) and optional ids (list[str])
        vectorstore.add_embeddings(
            embeddings=all_generated_embeddings,
            metadatas=all_chunk_metadatas,
            documents=all_chunk_texts # Pass the original texts corresponding to embeddings
            # Chroma will generate IDs if not provided
        )
        vectorstore.persist() # Make sure to persist changes
        st.write("--- Vector store created/updated ---")
    else:
        st.error("Mismatch between number of generated embeddings and splits. DB not fully populated.")
        raise ValueError("Embedding process failed for some chunks.")

    return vectorstore.as_retriever()
	

@st.cache_resource # Cache the entire RAG chain for a given URL
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

	#llm = ChatOpenAI(model=LLM_MODEL_NAME)
	# Use the Gemini 2.0 Flash model
	llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME) 


	history_aware_retriever = create_history_aware_retriever(
		llm, retriever, contextualize_q_prompt
	)

	# Example of how to create a QA prompt (ensure system_prompt is defined)
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
#os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
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
