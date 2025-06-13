import streamlit as st
import os
import glob
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
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
    """Loads, processes, and stores documents in a Chroma vector store."""
    
    persist_directory = "chroma_db_streamlit"
    
    if os.path.exists(persist_directory):
        st.info("Loading existing vector store from disk.")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=_embeddings
        )
        st.success("✅ Vector store loaded successfully from disk.")
        return vector_store

    st.info("Creating a new vector store. This is a one-time setup.")
    all_documents = []

    # --- Load JSON file from the provided data ---
    json_path = "discourse_posts.json" 
    if os.path.exists(json_path):
        # This jq_schema extracts the 'content' for the document body and other fields for metadata.
        # This is more robust than just '.[].content'.
        json_loader = JSONLoader(
            file_path=json_path,
            jq_schema='map({page_content: .content, metadata: {source: .url, author: .author, post_id: .post_id, topic_title: .topic_title}})',
            text_content=False,
        )
        try:
            json_documents = json_loader.load()
            st.write(f"Loaded {len(json_documents)} documents from '{json_path}'.")
            
            # --- DEBUG: Show a sample of the first loaded JSON document ---
            if json_documents:
                with st.expander("🕵️‍♀️ Click to view a sample loaded JSON document"):
                    st.json(json_documents[0].dict())

            all_documents.extend(json_documents)
        except Exception as e:
            st.error(f"Error loading or parsing {json_path}. Check the file format and jq_schema. Error: {e}")
    else:
        st.warning(f"'{json_path}' not found. Skipping Discourse posts.")

    if not all_documents:
        st.error("No documents were loaded. The application cannot proceed without data.")
        st.stop()

    # --- Split documents into chunks ---
    st.write(f"Splitting {len(all_documents)} documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(all_documents)
    st.write(f"Created {len(all_splits)} document chunks.")

    # --- Create and persist the Chroma vector store ---
    st.write("Creating vector store and generating embeddings... (this may take a moment)")
    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=_embeddings,
        persist_directory=persist_directory
    )
    
    st.success(f"✅ New vector store created and saved to disk with {len(all_splits)} chunks.")
    return vector_store


# --- 2. RAG LOGIC (LangGraph) ---

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def build_rag_graph(vector_store, llm, prompt):
    """Builds the LangGraph RAG chain."""
    
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
st.markdown("Ask a question about the 'Tools in Data Science' course. This AI assistant will answer based on course materials and forum discussions.")
st.markdown("---")

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
    user_question = st.text_area("❓ Your Question:", height=100, placeholder="e.g., Which subjects should I choose for the Jan term?")
    submitted = st.form_submit_button("Ask the Assistant")

if submitted and user_question:
    with st.spinner("Thinking... (Retrieving documents and generating answer)"):
        try:
            # Invoke the graph and get the final state
            response_state = rag_graph.invoke({"question": user_question})
            
            # Display the final answer
            st.markdown("### 💡 Answer")
            st.markdown(response_state["answer"])
            st.markdown("---")

            # --- DEBUG: Show the retrieved documents ---
            with st.expander("🔍 View Retrieved Context (The information used to generate the answer)"):
                st.markdown("These are the top 4 document chunks retrieved from the vector store that were most similar to your question:")
                for i, doc in enumerate(response_state["context"]):
                    st.info(f"**Document {i+1}:**\n\n" + doc.page_content)
                    st.json(doc.metadata, expanded=False)

            # --- DEBUG: Show the final prompt sent to the LLM ---
            with st.expander("📝 View Final Prompt Sent to LLM"):
                context_for_prompt = "\n\n".join(doc.page_content for doc in response_state["context"])
                final_prompt = rag_prompt.format(question=user_question, context=context_for_prompt)
                st.text(final_prompt)

        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("**About:** This is a demo of a RAG-based virtual TA for the IIT Madras 'Tools in Data Science' course.")
st.sidebar.markdown("**Powered by:**")
st.sidebar.markdown("- Streamlit\n- LangChain & LangGraph\n- Groq (LLaMA 3.1)\n- ChromaDB")
