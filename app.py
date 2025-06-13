import streamlit as st
import os
import glob
from typing import List, TypedDict
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from langchain.vectorstores import Chroma
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection

# --- 1. CONFIGURATION AND SETUP (Cached for Performance) ---

st.set_page_config(page_title="TDS Virtual Assistant", layout="wide")

@st.cache_resource
def configure_llm():
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.1-8b-instant",
            temperature=0.7,
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM. Set GROQ_API_KEY in secrets. Error: {e}")
        st.stop()

@st.cache_resource
def configure_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def load_vector_store_from_connection():
    configuration = {
        "client": "PersistentClient",
        "path": "./chroma_db"  # Same path used locally and on deployment
    }

    conn = st.connection("chromadb", type=ChromadbConnection, **configuration)
    try:
        st.success("Connected to ChromaDB successfully.")
        return Chroma(
            collection_name="documents_collection",  # Make sure this exists in local chroma_db
            embedding_function=configure_embeddings(),
            persist_directory="./chroma_db"
        )
    except Exception as e:
        st.error(f"Failed to connect Chroma vector store. Error: {e}")
        st.stop()

# --- 2. RAG LOGIC ---

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def build_rag_graph(vector_store, llm, prompt):
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"], k=4)
        return {"context": retrieved_docs}

    def generate(state: State):
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
st.markdown("Ask a question about the 'Tools in Data Science' course. This assistant answers based on course materials and forum posts.")

llm = configure_llm()
vector_store = load_vector_store_from_connection()

try:
    rag_prompt = hub.pull("rlm/rag-prompt")
except Exception as e:
    st.error(f"Could not pull prompt from LangChain Hub. Error: {e}")
    st.stop()

rag_graph = build_rag_graph(vector_store, llm, rag_prompt)

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
st.sidebar.markdown("**About:** Demo of a RAG-based TA for the IIT Madras 'Tools in Data Science' course.")
st.sidebar.markdown("**Powered by:**")
st.sidebar.markdown("- Streamlit\n- LangChain & LangGraph\n- Groq (LLaMA 3.1)\n- ChromaDB via st.connection")
