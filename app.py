
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
import PyPDF2
import uuid
# Load environment variables
load_dotenv()
# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
class SimpleModelSelector:
    """Simple class to handle model selection"""
    def __init__(self):
        self.llm_models = {
            "ollama": "Llama3.2"
        }
        self.embedding_models = {
            "chroma": {
                "name": "Chroma Default",
                "dimensions": 384,
                "model_name": None,
            },
        }
    def select_models(self):
        st.sidebar.title(":books: Model")
        llm = st.sidebar.radio(
            "Choose LLM Model:",
            options=list(self.llm_models.keys()),
            format_func=lambda x: self.llm_models[x],
        )
        embedding = st.sidebar.radio(
            "Choose Embedding Model:",
            options=list(self.embedding_models.keys()),
            format_func=lambda x: self.embedding_models[x]["name"],
        )
        return llm, embedding
class SimplePDFProcessor:
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    def read_pdf(self, pdf_file):
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    def create_chunks(self, text, pdf_file):
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if start > 0:
                start -= self.chunk_overlap
            chunk = text[start:end]
            if end < len(text):
                last_period = chunk.rfind(".")
                if last_period != -1:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "metadata": {"source": pdf_file.name},
            })
            start = end
        return chunks
class SimpleRAGSystem:
    def __init__(self, embedding_model="chroma", llm_model="ollama"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.db = chromadb.PersistentClient(path="./chroma_db")
        self.setup_embedding_function()
        self.collection = self.setup_collection()
    def setup_embedding_function(self):
        try:
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            st.error(f"Error setting up embedding function: {str(e)}")
            raise e
    def setup_collection(self):
        collection_name = f"documents_{self.embedding_model}"
        try:
            try:
                collection = self.db.get_collection(
                    name=collection_name, embedding_function=self.embedding_fn
                )
                st.info(f"Using existing collection for {self.embedding_model} embeddings")
            except:
                collection = self.db.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"model": self.embedding_model},
                )
                st.success(f"Created new collection for {self.embedding_model} embeddings")
            return collection
        except Exception as e:
            st.error(f"Error setting up collection: {str(e)}")
            raise e
    def add_documents(self, chunks):
        try:
            if not self.collection:
                self.collection = self.setup_collection()
            self.collection.add(
                ids=[chunk["id"] for chunk in chunks],
                documents=[chunk["text"] for chunk in chunks],
                metadatas=[chunk["metadata"] for chunk in chunks],
            )
            return True
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            return False
    def query_documents(self, query, n_results=3):
        try:
            if not self.collection:
                raise ValueError("No collection available")
            results = self.collection.query(query_texts=[query], n_results=n_results)
            return results
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return None
    def generate_response(self, query, context):
        try:
            import requests
            prompt = f"""
            Based on the following context, answer the question:
            Context: {context}
            Question: {query}
            Answer:
            """
            response = requests.post(
                "http://localhost:11434/v1/chat/completions",
                json={
                    "model": "llama3.2",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None
    def get_embedding_info(self):
        model_selector = SimpleModelSelector()
        model_info = model_selector.embedding_models[self.embedding_model]
        return {
            "name": model_info["name"],
            "dimensions": model_info["dimensions"],
            "model": self.embedding_model,
        }

    
def main():
        st.set_page_config(page_title="Ragrithm", layout="wide")
    
    # Dark theme styling
        st.markdown("""
    <style>
    body {
        background-color: #000000;
        color: white;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4, h5 {
        color: #00ffae;
    }
    .stButton>button {
        background-color: #00ffae;
        color: black;
        border: none;
        padding: 8px 16px;
        border-radius: 8px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    # Session state for landing screen
        if "started" not in st.session_state:
            st.session_state.started = False

        if not st.session_state.started:
            st.markdown("""
        <style>
        .landing-container {
            text-align: center;
            padding-top: 50px;
            padding-bottom: 30px;
        }
        .landing-heading {
            font-size: 40px;
            font-weight: bold;
            color: #00ffae;
        }
        .landing-subtext {
            font-size: 20px;
            color: white;
            margin-top: 10px;
            margin-bottom: 40px;
        }
        .robot-bg {
            background-image: url('https://cdn.pixabay.com/photo/2024/01/17/16/31/ai-8514972_1280.jpg'); /* fallback in case local image fails */
            background-size: cover;
            background-position: center;
            padding: 100px 30px;
            border-radius: 20px;
        }
        </style>
        <div class="robot-bg">
            <div class="landing-container">
                <div class="landing-heading">Welcome to Ragrithm !!</div>
                <div class="landing-subtext">Unlock deep insights from your documents using AI-powered PDF understanding.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
            st.markdown("""
        <div style='text-align: center; margin-top: -175px;'>
            <img src="https://static.vecteezy.com/system/resources/thumbnails/035/929/564/small_2x/robot-on-laptop-screen-talking-to-customer-automated-message-response-system-concept-free-video.jpg" width='700' alt='AI PDF Bot'/>
        </div>
    """, unsafe_allow_html=True)


            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Start Now", use_container_width=True):
                    st.session_state.started = True
                    st.rerun()

    # Move the features section here
            # st.markdown("---")
            # st.markdown("<h3 style='text-align:center;'>Key Features</h3>",unsafe_allow_html=True)
            # col1, col2, col3 = st.columns(3)
            # with col1:
            #     st.image("https://www.svgrepo.com/show/484943/pdf-file.svg", width=80)
            #     st.markdown("**Fast PDF Parsing**\n\nProcess and chunk large documents quickly.")
            # with col2:
            #     st.image("https://www.svgrepo.com/show/406078/letter-q.svg", width=80)
            #     st.markdown("**Smart Querying**\n\nAsk natural questions and get accurate answers.")
            # with col3:
            #     st.image("https://www.svgrepo.com/show/439117/content-security-policy.svg", width=80)
            #     st.markdown("**Context-Aware LLM**\n\nCombines retrieval + generation using Llama3.")
    
            return



    # ========== Main App Interface (after Start Now) ==========
        st.title("Ragrithm - Your Intelligent PDF Assistant")
        st.markdown("<div style='text-align: center; margin-top: 10px;'>", unsafe_allow_html=True)
        st.image("44.jpg", width=1200)
        st.markdown("</div>", unsafe_allow_html=True)


        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()
        if "current_embedding_model" not in st.session_state:
            st.session_state.current_embedding_model = None
        if "rag_system" not in st.session_state:
            st.session_state.rag_system = None

        model_selector = SimpleModelSelector()
        llm_model, embedding_model = model_selector.select_models()

        if embedding_model != st.session_state.current_embedding_model:
            st.session_state.processed_files.clear()
            st.session_state.current_embedding_model = embedding_model
            st.session_state.rag_system = None
            st.warning("Embedding model changed. Please re-upload your documents.")

        try:
            if st.session_state.rag_system is None:
                st.session_state.rag_system = SimpleRAGSystem(embedding_model, llm_model)
            embedding_info = st.session_state.rag_system.get_embedding_info()
            st.sidebar.info(
                f"Current Embedding Model:\n"
                f"- Name: {embedding_info['name']}\n"
                f"- Dimensions: {embedding_info['dimensions']}"
        )
        except Exception as e:
            st.error(f"Error initializing RAG system: {str(e)}")
            return

        pdf_file = st.file_uploader("📄 Upload PDF Document", type="pdf")
        if pdf_file and pdf_file.name not in st.session_state.processed_files:
            processor = SimplePDFProcessor()
            with st.spinner("🔍 Processing PDF..."):
                try:
                    text = processor.read_pdf(pdf_file)
                    chunks = processor.create_chunks(text, pdf_file)
                    if st.session_state.rag_system.add_documents(chunks):
                        st.session_state.processed_files.add(pdf_file.name)
                        st.success(f"✅ Successfully processed {pdf_file.name}")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

        if st.session_state.processed_files:
            st.markdown("---")
            st.subheader(":mag: Ask Questions About Your Document")
            query = st.text_input("Type your question here:")
            if query:
                with st.spinner("🤖 Generating answer..."):
                    results = st.session_state.rag_system.query_documents(query)
                    if results and results["documents"]:
                        response = st.session_state.rag_system.generate_response(
                            query, results["documents"][0]
                        )
                        if response:
                            st.markdown("### :memo: Answer:")
                            st.write(response)
                            with st.expander("📚 View Source Passages"):
                                for idx, doc in enumerate(results["documents"][0], 1):
                                    st.markdown(f"**Passage {idx}:**")
                                    st.info(doc)
        else:
            st.info(":point_up_2: Please upload a PDF to begin.")

    # ========== Features Section ==========
        # st.markdown("---")
        # st.subheader("✨ Key Features")
        # col1, col2, col3 = st.columns(3)
        # with col1:
        #     st.image("https://cdn-icons-png.flaticon.com/512/337/337946.png", width=80)
        #     st.markdown("**Fast PDF Parsing**\n\nProcess and chunk large documents quickly.")
        # with col2:
        #     st.image("https://cdn-icons-png.flaticon.com/512/3039/3039384.png", width=80)
        #     st.markdown("**Smart Querying**\n\nAsk natural questions and get accurate answers.")
        # with col3:
        #     st.image("https://cdn-icons-png.flaticon.com/512/1034/1034145.png", width=80)
        #     st.markdown("**Context-Aware LLM**\n\nCombines retrieval + generation using Llama3.")




if __name__ == "__main__":
    main()



  
