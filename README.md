#  DAVISS AI – Agentic RAG Assistant

DAVISS AI is an intelligent **Agentic Retrieval-Augmented Generation (RAG)** assistant developed using **Python, Streamlit, LangChain, LangGraph, ChromaDB, HuggingFace Embeddings, and Groq LLM**.

The application enables users to upload one or multiple PDF documents and interact with them through a conversational AI interface. It retrieves relevant information from the uploaded documents using semantic search and combines it with the reasoning capabilities of a Large Language Model (LLM) to generate accurate, context-aware responses.

---

#  Key Features

- 📄 Upload single or multiple PDF documents
- 🤖 Agentic workflow powered by LangGraph
- 🔍 Semantic document retrieval using ChromaDB
- 🧠 HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- ⚡ Groq LLM integration (`openai/gpt-oss-20b`)
- 💬 Interactive AI chatbot interface
- 📚 Answer questions from uploaded PDFs
- 🌍 Support for general knowledge questions
- 📑 Display retrieved document sources
- 📥 Download complete chat history
- 🎨 Modern dark-themed responsive UI
- 📱 Works on both desktop and mobile devices

---

#  Technologies Used

- Python
- Streamlit
- LangChain
- LangGraph
- ChromaDB
- HuggingFace Embeddings
- Groq API
- PyPDFLoader
- Recursive Character Text Splitter
- HTML & CSS (Custom Streamlit Styling)

---

#  How It Works

1. The user uploads one or more PDF documents.
2. The documents are loaded using **PyPDFLoader**.
3. Text is divided into smaller chunks using **RecursiveCharacterTextSplitter**.
4. Each chunk is converted into vector embeddings using **HuggingFace Embeddings**.
5. The embeddings are stored in **ChromaDB**.
6. When a user asks a question, the retriever searches for the most relevant document chunks.
7. LangGraph manages the agent workflow and decides when to use the retrieval tool.
8. Groq LLM generates a final response using the retrieved context.
9. The response is displayed along with the relevant document sources.

---

#  Project Structure

```
DAVISS-AI/
│
├── app.py
├── requirements.txt
├── README.md
└── sample.pdf
```

---

#  Installation

### Clone the repository

```bash
git clone https://github.com/your-username/DAVISS-AI.git
cd DAVISS-AI
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Create a `.env` file

```env
GROQ_API_KEY=your_groq_api_key
```

### Run the application

```bash
streamlit run app.py
```

---

#  Usage

- Launch the application.
- Upload one or more PDF documents in text format.
- Wait until the indexing process completes.
- Ask questions related to the uploaded documents.
- View retrieved document sources.
- Download the conversation history if required.

---

#  Future Improvements

- User Authentication
- Conversation Memory
- Voice-based Interaction
