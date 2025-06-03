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

🖼️ Landing Page:  ![6ddea8fa-fec2-4b27-b4bc-94b1e0acb926](https://github.com/user-attachments/assets/fdee1db0-ebfa-48bc-9d94-3ec0bf37d0c1)


📂 Upload PDF Interface : ![24506ca8-bef6-4eb9-9dc7-8da18179ffd6](https://github.com/user-attachments/assets/9f88c7b3-87be-4707-a435-6d3858bdfc34)


📂 PDF Uploaded: ![db370fe3-903e-43b1-84c8-a8c59e4b25d2](https://github.com/user-attachments/assets/61d626d0-322e-496b-b561-1c3aa8350350)


💬 Ask Questions: ![c13560ca-7ff2-4d70-9644-65ecaa64f177](https://github.com/user-attachments/assets/dc014566-e2a5-4dbc-b49f-37f9232b6df2)


📋 Answers & Source Context: ![bb801404-65c7-4ffc-b8c9-2fe9ca83629d](https://github.com/user-attachments/assets/fa2c2658-cc9b-438d-89c1-253f60cf76d5)


<h3>📦 Project Structure</h3>

├── app.py                  # Main Streamlit application

├── chroma_db/              # Local ChromaDB storage

├── screenshots/            # Screenshot images

├── requirements.txt        # Python dependencies

└── README.md               # This file




