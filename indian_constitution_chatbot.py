import os
import streamlit as st
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import StorageContext, VectorStoreIndex

# Set API keys (ensure these are properly secured)
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
os.environ["PINECONE_API_KEY"] = "YOUR_PINECONE_API_KEY"

# Initialize Models
def initialize_models():
    llm = Gemini()
    embed_model = GeminiEmbedding(model_name="models/embedding-001")
    return llm, embed_model

# Initialize Pinecone
def initialize_pinecone_client():
    from pinecone import Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    if api_key:
        return Pinecone(api_key=api_key)
    else:
        raise EnvironmentError("Pinecone API key not found in environment variables.")

# Create Vector Store and Storage Context
def create_vector_store_and_context(pinecone_index):
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return vector_store, storage_context

# Create Index from Documents
def create_index_from_documents(documents, storage_context):
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Streamlit App
st.title("Indian Constitution Chatbot")

# Initialize models and Pinecone
llm, embed_model = initialize_models()
pinecone_client = initialize_pinecone_client()
pinecone_index = pinecone_client.Index("knowledgeagent")

# Load documents (Ensure 'data/constitution.pdf' is uploaded beforehand)
if "index" not in st.session_state:
    from llama_index.core import SimpleDirectoryReader
    documents = SimpleDirectoryReader("data").load_data()
    vector_store, storage_context = create_vector_store_and_context(pinecone_index)
    index = create_index_from_documents(documents, storage_context)
    st.session_state.index = index

# Chatbot Interaction
user_input = st.text_input("Enter your query:", "")
if user_input:
    try:
        chat_engine = st.session_state.index.as_chat_engine()
        response = chat_engine.chat(user_input)
        st.write(f"**Chatbot:** {response.response}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


   
            
