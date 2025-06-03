<h1>Ragrithm – AI-Powered PDF Assistant</h1>

Ragrithm is a simple and powerful AI assistant that helps you understand and extract deep insights from your PDF documents using retrieval-augmented generation (RAG) and LLMs like Llama 3.2

<h3>🚀 Features:</h3>

📄 Fast PDF processing and chunking

🤖 Smart question-answering from documents

🔍 Context-aware responses with LLMs

🎯 Clean, dark-themed Streamlit UI

<h3>🛠️ Installation:</h3>

> Clone the repository:
git clone https://github.com/Anush-A/ragrithm.git
cd ragrithm

> Set up a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

> Install dependencies:
pip install -r requirements.txt

> Start Ollama (LLM backend) Make sure you have Ollama installed and running:
ollama run llama3

> Run the app
streamlit run app.py

<h3>📸 Screenshots:</h3>

🖼️ Landing Page:  <img width="1728" alt="Screenshot 2025-04-10 at 12 59 31 AM" src="https://github.ibm.com/Anush-A/llm-learning/assets/500280/6ddea8fa-fec2-4b27-b4bc-94b1e0acb926">

📂 Upload PDF Interface : <img width="1728" alt="Screenshot 2025-04-10 at 10 08 24 AM" src="https://github.ibm.com/Anush-A/llm-learning/assets/500280/24506ca8-bef6-4eb9-9dc7-8da18179ffd6">


📂 PDF Uploaded: <img width="1722" alt="Screenshot 2025-04-10 at 10 08 57 AM" src="https://github.ibm.com/Anush-A/llm-learning/assets/500280/db370fe3-903e-43b1-84c8-a8c59e4b25d2">

💬 Ask Questions: <img width="1728" alt="Screenshot 2025-04-10 at 10 09 20 AM" src="https://github.ibm.com/Anush-A/llm-learning/assets/500280/c13560ca-7ff2-4d70-9644-65ecaa64f177">


📋 Answers & Source Context: <img width="1720" alt="Screenshot 2025-04-10 at 10 09 41 AM" src="https://github.ibm.com/Anush-A/llm-learning/assets/500280/bb801404-65c7-4ffc-b8c9-2fe9ca83629d">

<h3>📦 Project Structure</h3>

├── app.py                  # Main Streamlit application

├── chroma_db/              # Local ChromaDB storage

├── screenshots/            # Screenshot images

├── requirements.txt        # Python dependencies

└── README.md               # This file




