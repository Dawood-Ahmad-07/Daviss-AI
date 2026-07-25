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
from langchain_core.messages import SystemMessage

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
html,body,[class*="css"]{font-family:'Poppins',sans-serif;}
.stApp{background:linear-gradient(180deg,#020617,#081224,#0f172a);color:white;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:2rem;max-width:1200px;}
section[data-testid="stSidebar"]{background:#050816 !important;border-right:1px solid #1e3a8a;}
.hero{text-align:center;margin-bottom:30px;}
.hero h1{color:white;font-size:58px;font-weight:800;margin-bottom:10px;}
.hero span{color:#4F7DFF;}
.hero p{color:#94a3b8;font-size:18px;}
.welcome{background:#0b1220;border:1px solid rgba(79,125,255,.25);border-radius:20px;padding:30px;margin-bottom:25px;}
[data-testid="stChatMessage"]{background:#101827;border-radius:18px;border:1px solid rgba(79,125,255,.15);padding:15px;}
.stChatInput{border:2px solid #4F7DFF !important;border-radius:18px !important;}
.stButton>button{background:linear-gradient(90deg,#2563eb,#4f46e5);color:white;border:none;border-radius:12px;width:100%;}
[data-testid="stFileUploader"]{background:#0f172a;border:2px dashed #4F7DFF;border-radius:18px;padding:15px;}
[data-testid="stChatMessage"] p,[data-testid="stChatMessage"] li,[data-testid="stChatMessage"] span,[data-testid="stChatMessage"] div{color:#ffffff !important;font-size:17px;font-weight:500;}
[data-testid="stChatMessage"] h1,[data-testid="stChatMessage"] h2,[data-testid="stChatMessage"] h3{color:#ffffff !important;}
[data-testid="stMarkdownContainer"] table td,[data-testid="stMarkdownContainer"] table th{color:white !important;}
table{width:100%;border-collapse:collapse;}
th{background:#1d4ed8;color:white;padding:10px;}
td{background:#111827;color:white;padding:10px;border:1px solid #374151;}
[data-testid="stChatInput"] textarea,[data-testid="stChatInput"] input{color:#ffffff !important;-webkit-text-fill-color:#ffffff !important;background:#111827 !important;font-size:18px !important;font-weight:500 !important;opacity:1 !important;}
[data-testid="stChatInput"] textarea::placeholder,[data-testid="stChatInput"] input::placeholder{color:#cbd5e1 !important;opacity:1 !important;}
[data-testid="stChatInput"]{background:#111827 !important;border:2px solid #2563eb !important;border-radius:18px !important;}
[data-testid="stChatInput"] textarea{caret-color:#4F7DFF !important;}
</style>
""", unsafe_allow_html=True)

# ====================================
# HERO
# ====================================

st.markdown("""
<div class="hero">
<div style="font-size:60px;">🤖</div>
<h1><span>DAVISS</span> AI</h1>
<p>
📄 Upload PDfs in text format &nbsp;&nbsp;•&nbsp;&nbsp;
💬 Ask Questions &nbsp;&nbsp;•&nbsp;&nbsp;
🌍 PDF + General Knowledge
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="welcome">
<h2 style="color:white;">✨ Welcome!</h2>
<p style="color:#cbd5e1;font-size:18px;">
Upload your documents and start asking questions.<br><br>
I can answer from your uploaded PDFs and also general knowledge.
</p>
</div>
""", unsafe_allow_html=True)

# ====================================
# 2. CACHED RESOURCES (sirf EK BAAR chalte hain, har rerun pe nahi)
# ====================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0)

@st.cache_resource
def init_vectorstore():
    """
    Sample.pdf ab load nahi hoga. Sirf empty vectorstore banega,
    jo user ki upload ki hui PDF se populate hoga.
    """
    embeddings = load_embeddings()
    persist_dir = "./chroma_db_new"

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore


# ====================================
# 3. HELPER FUNCTIONS (generic — kisi bhi PDF ke sath kaam karenge)
# ====================================

def get_dynamic_chunk_settings(documents):
    """
    PDF ka total size dekh kar chunk_size aur overlap khud decide karta hai.
    Chota PDF -> chota chunk (behtar precision).
    Bara PDF -> bara chunk (behtar context, kam chunks).
    """
    total_chars = sum(len(d.page_content) for d in documents)

    if total_chars < 5000:
        return 350, 40
    elif total_chars < 20000:
        return 500, 60
    elif total_chars < 60000:
        return 700, 100
    else:
        return 1000, 150


def clean_documents(documents):
    """
    Table-of-contents / index pages hata deta hai (jin mein dotted leaders
    jaise '......' ya bohot saari page-number listings hoti hain). Ye kisi
    bhi PDF ke liye generic hai, specific wording pe depend nahi karta.
    """
    cleaned = []
    for doc in documents:
        text = doc.page_content
        dot_count = text.count("...")
        # Agar page mein bohot dots hain ya bohot chhote fragments (numbers+dots)
        # bar bar aa rahe hain, ye TOC/index page lagta hai.
        if dot_count >= 3:
            continue
        cleaned.append(doc)
    return cleaned


def remove_overlap_duplicates(text):
    """
    Chunk overlap ki wajah se same sentence baar baar aa sakta hai.
    Ye function duplicate sentences hata deta hai, chahe content kuch bhi ho.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    result = []
    for s in sentences:
        s_clean = s.strip()
        key = s_clean.lower()
        if s_clean and key not in seen:
            seen.add(key)
            result.append(s_clean)
    return " ".join(result)


@st.cache_resource
def build_graph():
    llm = load_llm()
    vectorstore = init_vectorstore()

    @tool
    def retrieve_docs(query: str) -> str:
        """Search the user's uploaded PDF document(s) and return the most relevant
        text chunks, exactly as they appear in the document. You DO have access to
        the user's uploaded files through this tool — always call it whenever the
        user asks anything about their document, file, PDF, or its content, before
        answering. If it returns nothing relevant, say so explicitly and then fall
        back to general knowledge."""
        k = st.session_state.get("retriever_k", 4)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        if not docs:
            return "NOT_FOUND_IN_DOCUMENT"

        # Duplicate chunks hatao (overlap ki wajah se)
        seen = set()
        unique_parts = []
        for d in docs:
            content = d.page_content.strip()
            key = content.lower()
            if content and key not in seen:
                seen.add(key)
                unique_parts.append(content)

        combined = "\n\n".join(unique_parts)
        combined = remove_overlap_duplicates(combined)
        return combined

    tools = [retrieve_docs]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState):
        system_msg = SystemMessage(content=
                        (
        "You are an assistant with real access to the user's uploaded PDF "
        "documents through the retrieve_docs tool.\n\n"
        "FIRST CHECK: If the question is simple general knowledge, math, casual "
        "chat, or clearly has nothing to do with a document (e.g. '2+2', 'hello', "
        "'what is the capital of France'), answer it directly and briefly. Do NOT "
        "call retrieve_docs for such questions.\n\n"
        "Only for questions that could plausibly be about an uploaded document "
        "(the user references 'the document', 'the PDF', 'my file', asks about "
        "specific topics that might be explained in a document, etc.), follow "
        "these rules:\n"
        "1. Call retrieve_docs first before answering.\n"
        "2. If retrieve_docs returns relevant content (not 'NOT_FOUND_IN_DOCUMENT'), "
        "answer using the EXACT wording from the document as much as possible. Do "
        "NOT paraphrase or add your own examples.\n"
        "3. If retrieve_docs returns 'NOT_FOUND_IN_DOCUMENT', tell the user clearly: "
        "'Ye is document mein nahi mila.' Then optionally offer a general-knowledge "
        "answer, clearly labeled as 'General knowledge ke mutabiq: ...'.\n"
        "4. NEVER invent facts not present in retrieved text.\n"
        "5. Only answer exactly what was asked — trim any extra unrelated content "
        "from retrieved chunks.\n"
        "6. Ignore table-of-contents style text (dotted lines, bare page-number "
        "listings) in retrieved content.\n"
        "7. Never repeat duplicate sentences from retrieved content.\n\n"
        "Formatting rule: When presenting comparisons, structured data, or lists "
        "with multiple attributes, format them as a proper Markdown table:\n"
        "| Column1 | Column2 |\n|---------|---------|\n| value | value |\n\n"
        "Each row must be on its own line. Never put multiple rows on the same "
        "line separated by ||."
    ))
        response = llm_with_tools.invoke([system_msg] + state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


def fix_broken_table(text):
    """Fixes tables where LLM used || instead of newlines between rows"""
    text = re.sub(r'\|\s*\|', '|\n|', text)
    return text


# Initialize once, reused across reruns
vectorstore = init_vectorstore()
app = build_graph()


def add_pdf(path):
    """
    Naya PDF add karne se pehle purana sara data vectorstore se delete
    kar deta hai (sirf EK PDF hamesha indexed rehti hai). Chunk size PDF
    ke actual size ke hisaab se khud decide hoti hai, aur TOC/index pages
    automatically hata di jaati hain.
    """
    existing = vectorstore.get()
    if existing and existing.get("ids"):
        vectorstore.delete(ids=existing["ids"])

    loader = PyPDFLoader(path)
    docs = loader.load()

    # TOC / index jaisi pages hatao
    docs = clean_documents(docs)

    # Poora clean text session me save karo (raw/full view ke liye)
    st.session_state.full_pdf_text = "\n\n".join([d.page_content for d in docs])

    # PDF ke size ke hisaab se chunk size aur k khud decide karo
    chunk_size, chunk_overlap = get_dynamic_chunk_settings(docs)
    total_chars = sum(len(d.page_content) for d in docs)

    if total_chars < 5000:
        k_value = 3
    elif total_chars < 20000:
        k_value = 4
    elif total_chars < 60000:
        k_value = 6
    else:
        k_value = 8

    st.session_state.retriever_k = k_value

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    vectorstore.add_documents(chunks)

    return chunk_size, chunk_overlap, k_value

# ====================================
# STREAMLIT UI
# ====================================

st.title("🤖 Agentic RAG Assistant")
st.caption("📄 Upload PDFs • 💬 Ask Questions • 🌍 PDF + General Knowledge")

# ====================================
# SIDEBAR - UPLOAD (single PDF, replace on new upload)
# ====================================

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "retriever_k" not in st.session_state:
    st.session_state.retriever_k = 4

with st.sidebar:
    st.header("📂 Upload Document")

    uploaded_files = st.file_uploader(
        "Choose a PDF File",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Sirf sabse latest upload ki hui file rakhi jayegi
        latest_pdf = uploaded_files[-1]

        if st.session_state.current_file != latest_pdf.name:
            os.makedirs("uploaded_docs", exist_ok=True)
            save_path = os.path.join("uploaded_docs", latest_pdf.name)

            with open(save_path, "wb") as f:
                f.write(latest_pdf.getbuffer())

            with st.spinner("🔄 Indexing PDF..."):
                chunk_size, chunk_overlap, k_value = add_pdf(save_path)

            st.session_state.current_file = latest_pdf.name
            st.success(f"✅ {latest_pdf.name} Indexed (purani PDF replace ho gayi)")
            st.caption(f"Auto-tuned: chunk_size={chunk_size}, overlap={chunk_overlap}, k={k_value}")
        else:
            st.info(f"ℹ️ {latest_pdf.name} pehle se indexed hai")

    if st.session_state.get("full_pdf_text"):
        with st.expander("📖 Poori PDF ka Text Dekho"):
            st.text_area("Full Content", st.session_state.full_pdf_text, height=400)

    st.markdown("---")
    st.info(
        """
        🚀 Fast Agentic RAG

        📄 Single PDF at a time

        🌍 General Knowledge

        ⚡ Auto-tuned chunking

        🤖 Groq Powered
        """
    )

# ====================================
# CHAT MEMORY
# ====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# ====================================
# USER INPUT
# ====================================

prompt = st.chat_input("Ask anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        with st.spinner("🤖 Thinking..."):
            result = app.invoke({"messages": [("user", prompt)]})
            answer = result["messages"][-1].content
            answer = fix_broken_table(answer)

        streamed = ""
        for word in answer.split():
            streamed += word + " "
            placeholder.markdown(streamed + "▌", unsafe_allow_html=True)
            time.sleep(0.02)

        placeholder.markdown(streamed, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ====================================
# EXTRA FEATURES
# ====================================

with st.sidebar:
    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.success("🟢 Agent Online")
    st.caption("Powered by Groq + LangGraph + Chroma")
    st.caption("managed by Dawood Ahmad")

# ====================================
# DOCUMENT STATS
# ====================================

try:
    total_docs = vectorstore._collection.count()
    st.sidebar.metric("Indexed Chunks", total_docs)
except Exception:
    pass

# ====================================
# FOOTER
# ====================================

st.markdown("---")
st.caption("🚀 Agentic RAG | Single PDF | Groq | LangGraph | Chroma")

# ====================================
# SOURCE DOCUMENTS
# ====================================

with st.expander("📄 Retrieved Sources"):
    if prompt:
        k = st.session_state.get("retriever_k", 4)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
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

chat_text = ""
for m in st.session_state.messages:
    chat_text += f"{m['role'].upper()}:\n{m['content']}\n\n"

st.sidebar.download_button(
    "⬇ Download Chat",
    chat_text,
    file_name="chat_history.txt"
)