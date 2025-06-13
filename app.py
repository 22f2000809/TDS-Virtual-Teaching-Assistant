import streamlit as st
import os
import glob
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  # ✅ Replaced Chroma with FAISS
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
    """Loads, processes, and stores documents in a FAISS vector store."""
    
    all_documents = []

    # --- Load Markdown files ---
    markdown_files = glob.glob("./markdown_files/*.md")
    for file_path in markdown_files:
        loader = UnstructuredMarkdownLoader(file_path=file_path, mode="single", strategy="fast")
        documents = loader.load()
        all_documents.extend(documents)

    if not markdown_files:
        st.warning("No Markdown files found in './markdown_files/'. Add your notes.")

    # --- Load JSON file ---
    json_path = "discourse_posts.json"
    if os.path.exists(json_path):
        json_loader = JSONLoader(
            file_path=json_path,
            jq_schema='.[].content',
            text_content=False,
        )
        try:
            json_documents = json_loader.load()
            st.write(f"Loaded {len(json_documents)} documents from '{json_path}'.")

            if json_documents:
                with st.expander("🕵️ Sample JSON document"):
                    st.json(json_documents[0].dict())

            all_documents.extend(json_documents)
        except Exception as e:
            st.error(f"Error loading or parsing {json_path}. Error: {e}")
    else:
        st.warning(f"'{json_path}' not found. Skipping Discourse posts.")

    if not all_documents:
        st.error("No documents were loaded. Please check your inputs.")
        st.stop()

    # --- Split documents ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(all_documents)
    st.write(f"✅ {len(all_splits)} chunks created from source documents.")

    # --- Create FAISS vector store ---
    st.info("Creating FAISS index. This may take a moment.")
    vector_store = FAISS.from_documents(
        documents=all_splits,
        embedding=_embeddings
    )

    st.success("FAISS vector store is ready.")
    return vector_store

# --- 2. RAG LOGIC (LangGraph) ---

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def build_rag_graph(vector_store, llm, prompt):
    def retrieve(state: State):
        """Retrieves documents from the vector store."""
        retrieved_docs = vector_store.similarity_search(state["question"], k=4)
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
st.markdown("Ask a question about the 'Tools in Data Science' course. This assistant answers based on course materials and forum posts.")

llm = configure_llm()
embeddings = configure_embeddings()
vector_store = configure_vector_store(embeddings)

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
st.sidebar.markdown("- Streamlit\n- LangChain & LangGraph\n- Groq (LLaMA 3.1)\n- FAISS Vector Store")
