# ============ 1. IMPORTS ============
import os
import re
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="DAVISS AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================
# PREMIUM CSS
# ====================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html,body,[class*="css"]{
    font-family:'Poppins',sans-serif;
}

.stApp{
    background:linear-gradient(180deg,#020617,#081224,#0f172a);
    color:white;
}

header,footer,#MainMenu{
    visibility:hidden;
}

.block-container{
    padding-top:2rem;
    max-width:1200px;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#050816 !important;
    border-right:1px solid #1e3a8a;
}

/* Hero */
.hero{
    text-align:center;
    margin-bottom:30px;
}

.hero h1{
    color:white;
    font-size:58px;
    font-weight:800;
    margin-bottom:10px;
}

.hero span{
    color:#4F7DFF;
}

.hero p{
    color:#94a3b8;
    font-size:18px;
}

/* Welcome Card */
.welcome{
    background:#0b1220;
    border:1px solid rgba(79,125,255,.25);
    border-radius:20px;
    padding:30px;
    margin-bottom:25px;
}

/* Chat */
[data-testid="stChatMessage"]{
    background:#101827;
    border-radius:18px;
    border:1px solid rgba(79,125,255,.15);
    padding:15px;
}

/* Chat Input */
.stChatInput{
    border:2px solid #4F7DFF !important;
    border-radius:18px !important;
}

/* Button */
.stButton>button{
    background:linear-gradient(90deg,#2563eb,#4f46e5);
    color:white;
    border:none;
    border-radius:12px;
    width:100%;
}

/* File Uploader */
[data-testid="stFileUploader"]{
    background:#0f172a;
    border:2px dashed #4F7DFF;
    border-radius:18px;
    padding:15px;
}
/* Assistant aur User text */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div{
    color: #ffffff !important;
    font-size: 17px;
    font-weight: 500;
}

/* Markdown headings */
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3{
    color: #ffffff !important;
}

/* Tables */
[data-testid="stMarkdownContainer"] table td,
[data-testid="stMarkdownContainer"] table th{
    color: white !important;
}

/* Table */
table{
    width:100%;
    border-collapse:collapse;
}

th{
    background:#1d4ed8;
    color:white;
    padding:10px;
}

td{
    background:#111827;
    color:white;
    padding:10px;
    border:1px solid #374151;
}
/* ===========================
   FIX CHAT INPUT TEXT
=========================== */

/* Jo text tum type karte ho */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input{
    color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
    background:#111827 !important;
    font-size:18px !important;
    font-weight:500 !important;
    opacity:1 !important;
}

/* Placeholder */
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInput"] input::placeholder{
    color:#cbd5e1 !important;
    opacity:1 !important;
}

/* Input box */
[data-testid="stChatInput"]{
    background:#111827 !important;
    border:2px solid #2563eb !important;
    border-radius:18px !important;
}

/* Cursor */
[data-testid="stChatInput"] textarea{
    caret-color:#4F7DFF !important;
}
</style>
""", unsafe_allow_html=True)

# ====================================
# HERO
# ====================================

st.markdown("""
<div class="hero">

<div style="font-size:60px;">🤖</div>

<h1><span>Agentic RAG</span> Assistant</h1>

<p>
📄 Upload PDFs &nbsp;&nbsp;•&nbsp;&nbsp;
💬 Ask Questions &nbsp;&nbsp;•&nbsp;&nbsp;
🌍 PDF + General Knowledge
</p>

</div>
""", unsafe_allow_html=True)

# ====================================
# WELCOME CARD
# ====================================

st.markdown("""
<div class="welcome">

<h2 style="color:white;">✨ Welcome!</h2>

<p style="color:#cbd5e1;font-size:18px;">
Upload your documents and start asking questions.<br><br>
I can answer from your uploaded PDFs and also general knowledge.
</p>

</div>
""", unsafe_allow_html=True)
# ============ 2. LOAD PDF ============


from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample.pdf")
documents = loader.load()

print(os.getcwd())
print("Pages:", len(documents))

#chunking the document into smaller pieces for better retrieval
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Chunks:", len(chunks))

# ============ 3. EMBEDDINGS (LOCAL, FREE, NO LIMIT) ============
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# ============ 4. VECTORSTORE (FRESH EVERY TIME) ============
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_new"   # naya folder, purane wale se conflict nahi hoga
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
def add_pdf(path):
    loader = PyPDFLoader(path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    vectorstore.add_documents(chunks)

# ============ 5. LLM (GROQ) ============
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

def fix_broken_table(text):
    """Fixes tables where LLM used || instead of newlines between rows"""
    text = re.sub(r'\|\s*\|', '|\n|', text)
    return text


# ============ 6. RETRIEVER TOOL ============
@tool
def retrieve_docs(query: str) -> str:
    """Search the document and return relevant chunks in a proper and correct way. arranged it a proper manner."""
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])

tools = [retrieve_docs]
llm_with_tools = llm.bind_tools(tools)
# ============ 7. AGENT NODE ============
from langchain_core.messages import SystemMessage

def agent_node(state: MessagesState):
    system_msg = SystemMessage(content=(
        "When presenting comparisons, structured data, or lists with multiple attributes, "
        "always format them as a proper Markdown table using this exact syntax:\n"
        "| Column1 | Column2 |\n|---------|---------|\n| value | value |\n\n"
        "Each row must be on its own line. Never put multiple rows on the same line separated by ||."
    ))
    response = llm_with_tools.invoke([system_msg] + state["messages"])
    return {"messages": [response]}
# ============ 8. TOOL NODE ============
tool_node = ToolNode(tools)    
 # ============ 9. BUILD GRAPH ============
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")
app = graph.compile()
# ============ 10. TEST ============
result = app.invoke({"messages": [("user", "about document")]})

print(result["messages"][-1].content)
# ====================================
# STREAMLIT UI
# ====================================

st.title("🤖 Agentic RAG Assistant")

st.caption("📄 Upload PDFs • 💬 Ask Questions • 🌍 PDF + General Knowledge")

# ====================================
# SIDEBAR
# ====================================

with st.sidebar:

    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        progress = st.progress(0)

        total = len(uploaded_files)

        for i, pdf in enumerate(uploaded_files):

            save_path = os.path.join(
                "uploaded_docs",
                pdf.name
            )

            os.makedirs(
                "uploaded_docs",
                exist_ok=True
            )

            with open(save_path, "wb") as f:
                f.write(pdf.getbuffer())

            add_pdf(save_path)

            progress.progress(
                (i + 1) / total
            )

        st.success("✅ PDFs Indexed Successfully")

    st.markdown("---")

    st.info(
        """
        🚀 Fast Agentic RAG

        📄 Multiple PDF Support

        🌍 General Knowledge

        ⚡ Cached Vector Database

        🤖 Groq Powered
        """
    )

# ====================================
# CHAT MEMORY
# ====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ====================================
# DISPLAY CHAT
# ====================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ====================================
# USER INPUT
# ====================================

prompt = st.chat_input(
    "Ask anything..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    with st.chat_message("assistant"):

        placeholder = st.empty()

        with st.spinner("🤖 Thinking..."):

            result = app.invoke(
                {
                    "messages":[
                        ("user",prompt)
                    ]
                }
            )

            answer = result["messages"][-1].content
            answer = fix_broken_table(answer)   # 👈 YE LINE ADD KARO

        # Streaming Effect (character-based, safe)
        
        streamed = ""

        for word in answer.split():

            streamed += word + " "

            placeholder.markdown(streamed + "▌")

            time.sleep(0.02)

        placeholder.markdown(streamed)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )
    # ====================================
# EXTRA FEATURES
# ====================================

with st.sidebar:

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages=[]

        st.rerun()

    st.markdown("---")

    st.success("🟢 Agent Online")

    st.caption("Powered by Groq + LangGraph + Chroma")
    st.caption("managed by Dawood Ahmad ")

# ====================================
# DOCUMENT STATS
# ====================================

try:

    total_docs = vectorstore._collection.count()

    st.sidebar.metric(
        "Indexed Chunks",
        total_docs
    )

except:

    pass

# ====================================
# FOOTER
# ====================================

st.markdown("---")

st.caption(
    "🚀 Agentic RAG | Multi PDF | Groq | LangGraph | Chroma"
)

# ====================================
# SOURCE DOCUMENTS
# ====================================

with st.expander("📄 Retrieved Sources"):

    if prompt:

        docs = retriever.invoke(prompt)

        for i, doc in enumerate(docs):

            page = doc.metadata.get("page", "Unknown")

            source = doc.metadata.get("source", "Uploaded PDF")

            st.markdown(f"### Source {i+1}")

            st.write(f"📄 File : {source}")

            st.write(f"📑 Page : {page}")

            st.info(doc.page_content[:400] + "...")

# ====================================
# DOWNLOAD CHAT
# ====================================

chat_text=""

for m in st.session_state.messages:

    chat_text += f"{m['role'].upper()}:\n{m['content']}\n\n"

st.sidebar.download_button(

    "⬇ Download Chat",

    chat_text,

    file_name="chat_history.txt"

)