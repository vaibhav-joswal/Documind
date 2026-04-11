# app.py
import os
import json
import streamlit as st
from openai import OpenAI
from pdf import OptimizedMultiDocRAG  # Updated import
from pathlib import Path
from datetime import datetime

# ---------------- CONFIG ----------------
# Put your HF_ROUTER API key here or set env var HF_API_KEY
HF_API_KEY = os.getenv("HF_API_KEY") 
DOC_CACHE_DIR = "./doc_cache"  # Changed from pdf_cache to doc_cache
Path(DOC_CACHE_DIR).mkdir(exist_ok=True)

# OpenAI client (HuggingFace router)
client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_API_KEY)

# ---------------- PAGE ----------------
st.set_page_config(page_title="📄 Chat with Documents", page_icon="🤖", layout="wide")
st.markdown(
    "<h1 style='color:#1E88E5;'>🤖 Chat with Your Documents</h1>",
    unsafe_allow_html=True,
)
st.markdown("Upload documents (PDF, Word, TXT, CSV) in the sidebar, process them and ask questions. The bot will answer using the document context.")

# ---------------- STYLES (full unified CSS) ----------------
st.markdown(
    """
<style>
/* Sidebar (light & readable) */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-right: 2px solid #e6e6e6;
    padding-top: 12px;
}

/* Sidebar header styling */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #1E88E5 !important;
    font-weight: 700;
}

/* File uploader */
div[data-testid="stFileUploaderDropzone"] {
    background-color: #fafafa !important;
    border: 2px dashed #1E88E5 !important;
    border-radius: 10px;
    padding: 18px !important;
    color: #000 !important;
}
div[data-testid="stFileUploaderDropzone"] * {
    color: #000 !important;
}
div[data-testid="stFileUploaderFileName"], div[data-testid="stFileUploaderFileName"] * {
    color: #000 !important;
    font-weight: 600;
}

/* Document info box */
.doc-info {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 8px;
    margin: 8px 0;
    border-left: 4px solid #1E88E5;
}

/* File type badges */
.file-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 600;
    margin: 2px;
}
.badge-pdf { background-color: #ffebee; color: #c62828; }
.badge-docx { background-color: #e3f2fd; color: #1565c0; }
.badge-txt { background-color: #f3e5f5; color: #7b1fa2; }
.badge-csv { background-color: #e8f5e8; color: #2e7d32; }

/* Chat container (scrollable) */
.chat-container {
    max-height: 70vh;
    overflow-y: auto;
    padding-right: 12px;
    margin-bottom: 16px;
}

/* User bubble (right) */
.user-msg {
    background-color: #DCF8C6 !important;
    color: #000000 !important;
    padding: 10px 14px;
    border-radius: 14px;
    margin: 8px 0;
    display: inline-block;
    max-width: 80%;
    float: right;
    clear: both;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}

/* Bot bubble (left) */
.bot-msg {
    background-color: #C5CAE9 !important;
    color: #000000 !important;
    padding: 10px 14px;
    border-radius: 14px;
    margin: 8px 0;
    display: inline-block;
    max-width: 80%;
    float: left;
    clear: both;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}

/* clear floats */
.chat-container::after {
    content: "";
    display: table;
    clear: both;
}

/* Process button style */
div.stButton > button {
    background-color: #1E88E5 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-weight: 700 !important;
}
div.stButton > button:hover {
    background-color: #1565C0 !important;
}

/* Danger button for clear/remove */
.danger-button > button {
    background-color: #f44336 !important;
    color: white !important;
}
.danger-button > button:hover {
    background-color: #d32f2f !important;
}

/* small screens */
@media (max-width: 600px) {
    .user-msg, .bot-msg { max-width: 95% !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- HELPER FUNCTIONS ----------------
def get_file_badge(file_path):
    """Generate colored badge based on file extension"""
    ext = Path(file_path).suffix.lower()
    badges = {
        '.pdf': '<span class="file-badge badge-pdf">PDF</span>',
        '.docx': '<span class="file-badge badge-docx">DOCX</span>',
        '.doc': '<span class="file-badge badge-docx">DOC</span>',
        '.txt': '<span class="file-badge badge-txt">TXT</span>',
        '.csv': '<span class="file-badge badge-csv">CSV</span>'
    }
    return badges.get(ext, f'<span class="file-badge">{ext.upper()}</span>')

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"

# ---------------- SESSION STATE ----------------
if "rag" not in st.session_state:
    st.session_state.rag = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {time, question, answer}
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []  # list of file paths
if "processing_mode" not in st.session_state:
    st.session_state.processing_mode = "single"  # "single" or "multiple"

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### 📂 Upload & Process Documents")
    
    # Processing mode selection
    processing_mode = st.radio(
        "📋 Processing Mode:",
        ["single", "multiple"],
        format_func=lambda x: "Single Document" if x == "single" else "Multiple Documents",
        key="processing_mode"
    )
    
    if processing_mode == "single":
        # Single file upload
        uploaded_file = st.file_uploader(
            "📄 Choose a document",
            type=["pdf", "docx", "doc", "txt", "csv"],
            key="single_file"
        )
        
        if uploaded_file:
            file_path = os.path.join(DOC_CACHE_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.uploaded_files = [file_path]
            
            # Show file info
            file_size = format_file_size(len(uploaded_file.getbuffer()))
            st.markdown(
                f"""
                <div class="doc-info">
                    <strong>Uploaded:</strong><br>
                    {get_file_badge(file_path)} {uploaded_file.name}<br>
                    <small>Size: {file_size}</small>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    else:
        # Multiple file upload
        uploaded_files = st.file_uploader(
            "📄 Choose multiple documents",
            type=["pdf", "docx", "doc", "txt", "csv"],
            accept_multiple_files=True,
            key="multiple_files"
        )
        
        if uploaded_files:
            file_paths = []
            total_size = 0
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DOC_CACHE_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
                total_size += len(uploaded_file.getbuffer())
            
            st.session_state.uploaded_files = file_paths
            
            # Show files info
            st.markdown(
                f"""
                <div class="doc-info">
                    <strong>Uploaded {len(uploaded_files)} files:</strong><br>
                    {''.join([f"{get_file_badge(fp)} {Path(fp).name}<br>" for fp in file_paths])}
                    <small>Total size: {format_file_size(total_size)}</small>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Show current processed documents
    if st.session_state.rag and st.session_state.uploaded_files:
        st.markdown("---")
        st.markdown("**📚 Processed Documents:**")
        try:
            doc_info = st.session_state.rag.get_document_info()
            st.text(doc_info)
        except:
            for file_path in st.session_state.uploaded_files:
                st.markdown(f"- {get_file_badge(file_path)} {Path(file_path).name}", unsafe_allow_html=True)

    # Controls
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        process_button_text = "⚡ Process" if processing_mode == "single" else "⚡ Process All"
        if st.button(process_button_text) and st.session_state.uploaded_files:
            try:
                with st.spinner("Processing documents (this can take a moment)..."):
                    rag_system = OptimizedMultiDocRAG(
                        model_name="all-MiniLM-L6-v2",
                        cache_dir=DOC_CACHE_DIR,
                        max_words=150,
                        overlap=30,
                    )
                    
                    if processing_mode == "single":
                        rag_system.process_document(st.session_state.uploaded_files[0], force_reprocess=False)
                    else:
                        rag_system.process_multiple_documents(st.session_state.uploaded_files, force_reprocess=False)
                    
                    st.session_state.rag = rag_system
                    st.session_state.chat_history = []
                    st.success(f"📚 {len(st.session_state.uploaded_files)} document(s) processed successfully!")
                    
            except Exception as e:
                st.error(f"Error processing documents: {e}")
                
    with col2:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Additional controls
    st.markdown("---")
    col3, col4 = st.columns([1, 1])
    
    with col3:
        if st.button("🗑️ Clear All", key="clear_all"):
            st.session_state.uploaded_files = []
            st.session_state.rag = None
            st.session_state.chat_history = []
            # Clean up cache directory
            import shutil
            if os.path.exists(DOC_CACHE_DIR):
                shutil.rmtree(DOC_CACHE_DIR)
            Path(DOC_CACHE_DIR).mkdir(exist_ok=True)
            st.rerun()
    
    with col4:
        # Download chat history
        if st.session_state.chat_history:
            if st.button("⬇️ Export Chat"):
                chat_data = {
                    "export_time": datetime.now().isoformat(),
                    "processed_files": [Path(f).name for f in st.session_state.uploaded_files],
                    "chat_history": st.session_state.chat_history
                }
                fname = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(chat_data, ensure_ascii=False, indent=2),
                    file_name=fname,
                    mime="application/json"
                )

# ---------------- MAIN: chat area ----------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
# render chat history from session_state
for item in st.session_state.chat_history:
    q = item.get("question", "")
    a = item.get("answer", "")
    # user (right)
    if q:
        st.markdown(f'<div class="user-msg">👤 <b>You:</b> {q}</div>', unsafe_allow_html=True)
    if a:
        st.markdown(f'<div class="bot-msg">🤖 <b>Bot:</b> {a}</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# If no documents processed, show info
if not st.session_state.rag:
    if not st.session_state.uploaded_files:
        st.info("Please upload document(s) from the sidebar to start chatting.")
    else:
        st.info("Documents uploaded! Click 'Process' in the sidebar to enable chatting.")
else:
    # Chat input (shows at bottom)
    question = st.chat_input("💬 Ask a question about your documents...")
    if question:
        # 1) Append user message instantly (so it shows without delay)
        entry = {"time": datetime.now().isoformat(), "question": question, "answer": ""}
        st.session_state.chat_history.append(entry)

        # Force immediate display of user message
        st.rerun()

    # If last history item has empty answer and rag exists -> generate response (streaming)
    if st.session_state.chat_history:
        last = st.session_state.chat_history[-1]
        if last.get("answer", "") == "" and last.get("question", "") and st.session_state.rag:
            question = last["question"]

            # Create placeholder for streaming
            placeholder = st.empty()
            full_response = ""

            # Stream chunks from backend
            try:
                for chunk in st.session_state.rag.generate_response_stream(question):
                    full_response += chunk
                    placeholder.markdown(
                        f'<div class="bot-msg">🤖 <b>Bot:</b> {full_response}▌</div>',
                        unsafe_allow_html=True
                    )

                # Final render without cursor
                placeholder.markdown(
                    f'<div class="bot-msg">🤖 <b>Bot:</b> {full_response}</div>',
                    unsafe_allow_html=True
                )

                # Save final response in chat history
                st.session_state.chat_history[-1]["answer"] = full_response
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                placeholder.markdown(
                    f'<div class="bot-msg">🤖 <b>Bot:</b> {error_msg}</div>',
                    unsafe_allow_html=True
                )
                st.session_state.chat_history[-1]["answer"] = error_msg
                
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<small>💡 <strong>Tip:</strong> You can upload multiple document types (PDF, DOCX, TXT, CSV) and chat with them together!</small>",
    unsafe_allow_html=True
)