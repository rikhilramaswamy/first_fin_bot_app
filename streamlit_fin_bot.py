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
#sys.modules["sqlite3"] = pysqlite3
#import chromadb
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


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


# --- Configuration ---
CHUNK_SIZE = 700  # Smaller chunk size
CHUNK_OVERLAP = 100 # Adjusted overlap
BATCH_SIZE = 30   # Number of document contents to send per embedding API call
MAX_RETRIES = 5
INITIAL_BACKOFF_TIME = 1 # seconds


# Try importing from the main `google.generativeai` module itself,
# or from `google.generativeai.types` if that's where they are.
# A common pattern is to just import google.generativeai and access its types.
import google.generativeai as genai
# If the above doesn't work, then try:
# from google.generativeai.types import Content, Part
# Or if it's in protos (less common for direct use but possible):
# from google.generativeai import protos as genai_protos # Then use genai_protos.Content, genai_protos.Part


# --- Helper function for embedding with retry logic, now with proper caching ---
@st.cache_data(ttl=600, show_spinner="Generating embeddings...") # Cache for 10 minutes (600 seconds)
def get_embeddings_with_retry(
    model_name: str,             # Pass model name as string (hashable)
    content_batch: list[str],    # This is still a list of Python strings
    max_retries: int = MAX_RETRIES,
    initial_backoff: int = INITIAL_BACKOFF_TIME
) -> list[list[float]]:
    """
    Attempts to get embeddings for a batch of content with retry logic.
    This function is cached by Streamlit.
    """
    st.write(f"Embedding a batch of {len(content_batch)} documents...")

    temp_embedding_model = GoogleGenerativeAIEmbeddings(
        model=model_name,
        task_type="RETRIEVAL_DOCUMENT",
        embed_kwargs={"output_dimensionality": 512}
    )
    api_client = temp_embedding_model.client # Get the underlying client

    # --- Convert list[str] to list[Content] for the raw API client ---
    # Use genai.types.Content and genai.types.Part if direct import fails,
    # or just genai.Content and genai.Part if they are top-level.
    # Based on the latest SDK, `genai.types.Content` and `genai.types.Part`
    # or sometimes directly `genai.GenerationConfig` and `genai.HarmCategory` are used.
    # However, for constructing content, the `GenerativeModel`'s `generate_content`
    # method is designed to handle raw strings and convert them.
    # The direct `client.embed_content` still requires the protobuf types.
    # Let's try `genai.types.Content` and `genai.types.Part` as the most likely.
    api_content_batch = [
        genai.types.Content(parts=[genai.types.Part(text=text_item)]) for text_item in content_batch
    ]
    # --- END CONVERSION ---

    retries = 0
    backoff_time = initial_backoff

    while retries < max_retries:
        try:
            response = api_client.embed_content(
                model=model_name,
                content=api_content_batch
            )
            return [e.values for e in response.embeddings]
        except exceptions.ServiceUnavailable as e:
            st.warning(f"Service Unavailable (503) during embedding, retrying in {backoff_time}s... ({e})")
        except exceptions.DeadlineExceeded as e:
            st.warning(f"Deadline Exceeded (504) during embedding, retrying in {backoff_time}s... ({e})")
        except exceptions.ResourceExhausted as e:
            st.warning(f"Rate Limit Exceeded (429) during embedding, retrying in {backoff_time}s... ({e})")
            backoff_time = min(backoff_time * 3, 60)
        except Exception as e:
            st.error(f"An unexpected error occurred during embedding: {e}")
            raise

        time.sleep(backoff_time)
        backoff_time *= 2
        retries += 1

    raise Exception(f"Failed to get embeddings after {max_retries} retries.")



@st.cache_resource  # Cache the creation of vector store if documents are processed in-app
# --- Main vector_retriever function (remains mostly the same) ---
def vector_retriever(_docs: list[Document]):
    st.write("--- Inside vector_retriever function ---")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(_docs)
    st.write(f"Split {len(_docs)} documents into {len(splits)} chunks.")

    model_name = "models/text-embedding-004"

    persistent_db_path = os.path.join(os.getcwd(), "mydb.chromadb")
    os.makedirs(persistent_db_path, exist_ok=True)

    gemini_embeddings_for_chroma = GoogleGenerativeAIEmbeddings(
        model=model_name,
        task_type="RETRIEVAL_DOCUMENT",
        embed_kwargs={"output_dimensionality": 512}
    )

    vectorstore = Chroma(
        persist_directory=persistent_db_path,
        embedding_function=gemini_embeddings_for_chroma
    )

    st.write(f"Starting embedding process with batch size: {BATCH_SIZE}...")

    all_chunk_texts = [s.page_content for s in splits]
    all_chunk_metadatas = [s.metadata for s in splits]

    all_generated_embeddings = []

    for i in range(0, len(all_chunk_texts), BATCH_SIZE):
        chunk_batch_texts = all_chunk_texts[i : i + BATCH_SIZE]

        try:
            batch_embeddings = get_embeddings_with_retry(
                model_name=model_name,
                content_batch=chunk_batch_texts
            )
            all_generated_embeddings.extend(batch_embeddings)
            st.write(f"Successfully embedded batch {i // BATCH_SIZE + 1} of {len(all_chunk_texts) // BATCH_SIZE + 1}")
        except Exception as e:
            st.error(f"Failed to embed batch {i // BATCH_SIZE + 1} after retries. Error: {e}")
            raise

    if len(all_generated_embeddings) == len(splits):
        st.write(f"Adding {len(all_generated_embeddings)} embeddings to ChromaDB...")
        vectorstore.add_embeddings(
            embeddings=all_generated_embeddings,
            metadatas=all_chunk_metadatas,
            documents=all_chunk_texts
        )
        vectorstore.persist()
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
