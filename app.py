import streamlit as st
import os
import glob
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langgraph.graph import START, StateGraph
from typing import List, TypedDict
from langchain_core.documents import Document

# --- 1. CONFIGURATION AND SETUP (Cached for Performance) ---

st.set_page_config(page_title="TDS Virtual Assistant", layout="wide")

@st.cache_resource
def configure_llm():
    """Initializes and caches the Groq LLM."""
    try:
        # Use Streamlit's secrets management
        groq_api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.1-8b-instant",
            temperature=0.7,
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize the LLM. Have you set your GROQ_API_KEY in Streamlit secrets? Error: {e}")
        st.stop()

@st.cache_resource
def configure_embeddings():
    """Initializes and caches the sentence-transformer embeddings."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource(show_spinner="Loading and indexing documents... This may take a minute.")
def configure_vector_store(_embeddings):
    """Loads, processes, and stores documents in a Chroma vector store."""
    
    # Check if the vector store has already been created and persisted
    persist_directory = "chroma_db_streamlit"
    if os.path.exists(persist_directory):
        st.info("Loading existing vector store from disk.")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=_embeddings
        )
        return vector_store

    st.info("Creating new vector store. This is a one-time setup.")
    # --- Load all Markdown files ---
    markdown_files = glob.glob("./markdown_files/*.md")
    if not markdown_files:
        st.warning("No Markdown files found in './markdown_files/'. Please add your scraped data.")
    
    all_documents = []
    for file_path in markdown_files:
        loader = UnstructuredMarkdownLoader(file_path=file_path, mode="single", strategy="fast")
        documents = loader.load()
        all_documents.extend(documents)

    # --- Load JSON file ---
    json_path = "discourse_posts.json"
    if os.path.exists(json_path):
        json_loader = JSONLoader(
            file_path=json_path,
            jq_schema=".[].content", # Assumes a list of objects with a 'content' key
            text_content=False,
        )
        json_documents = json_loader.load()
        all_documents.extend(json_documents)
    else:
        st.warning(f"'{json_path}' not found. Skipping Discourse posts.")

    if not all_documents:
        st.error("No documents were loaded. The application cannot proceed without data.")
        st.stop()

    # --- Split documents into chunks ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(all_documents)

    # --- Create and persist the Chroma vector store ---
    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=_embeddings,
        persist_directory=persist_directory  # Persist the DB to disk
    )
    
    st.success(f"Vector store created with {len(all_splits)} document chunks.")
    return vector_store


# --- 2. RAG LOGIC (LangGraph) ---

# Define state for the graph
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def build_rag_graph(vector_store, llm, prompt):
    """Builds the LangGraph RAG chain."""
    
    def retrieve(state: State):
        """Retrieves documents from the vector store."""
        retrieved_docs = vector_store.similarity_search(state["question"], k=4) # Retrieve top 4 docs
        return {"context": retrieved_docs}

    def generate(state: State):
        """Generates an answer using the retrieved context."""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    
    return graph_builder.compile()

# --- 3. STREAMLIT UI ---

st.title("🎓 TDS Virtual Teaching Assistant")
st.markdown("Ask a question about the 'Tools in Data Science' course. This AI assistant will answer based on course materials and forum discussions.")

# Initialize all components
llm = configure_llm()
embeddings = configure_embeddings()
vector_store = configure_vector_store(embeddings)

# Pull the RAG prompt from LangChain Hub
try:
    rag_prompt = hub.pull("rlm/rag-prompt")
except Exception as e:
    st.error(f"Could not pull prompt from LangChain Hub. Please check your network connection. Error: {e}")
    st.stop()

# Build the RAG graph
rag_graph = build_rag_graph(vector_store, llm, rag_prompt)

# Create a form for the user input
with st.form("question_form"):
    user_question = st.text_area("❓ Your Question:", height=100)
    submitted = st.form_submit_button("Ask the Assistant")

if submitted and user_question:
    with st.spinner("Thinking..."):
        try:
            response = rag_graph.invoke({"question": user_question})
            st.markdown("### 💡 Answer")
            st.markdown(response["answer"])
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("**About:** This is a demo of a RAG-based virtual TA for the IIT Madras 'Tools in Data Science' course.")
st.sidebar.markdown("**Powered by:**")
st.sidebar.markdown("- Streamlit\n- LangChain & LangGraph\n- Groq (LLaMA 3.1)\n- ChromaDB")