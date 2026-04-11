# 📄 Universal Multi-Document AI Chat (RAG)

Welcome to the **Universal Multi-Document RAG Chatbot**! This is a powerful, locally-hosted Streamlit application that allows you to upload multiple documents (PDFs, Word Docs, Text Files, and CSVs), build a highly optimized searchable database out of them, and chat with an AI assistant that answers your questions based *only* on the context of your uploaded files.

## 🚀 Features

- **Multi-Format Support:** Seamlessly process `.pdf`, `.docx`, `.txt`, and `.csv` files.
- **Advanced Text Extraction & OCR:** Utilizes `PyMuPDF` for blazing-fast native text extraction, and automatically falls back to `PyTesseract` (OCR) for scanned PDFs and images.
- **Lightning Fast Retrieval:** Uses state-of-the-art **FAISS** (Facebook AI Similarity Search) and `SentenceTransformers` (`all-MiniLM-L6-v2`) to chunk, embed, and search your documents in under 10 ms.
- **Intelligent Chunking:** Smart text splitting that tracks source files and page numbers for perfect source attribution.
- **Streaming Chat UI:** A modern, scrolling chat interface built with Streamlit that visually streams the AI's thought process and responses in real-time.
- **Local Caching:** Automatically caches embedded documents. Restarting the server won't force you to wait for re-processing!

## 🏗️ Technology Stack

- **Frontend:** Streamlit 
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Database:** `faiss-cpu`
- **Text Extraction:** `PyMuPDF` (fitz), `python-docx`, `pandas`
- **OCR:** `pytesseract`, `pdf2image`, `opencv-python-headless`
- **LLM Engine:** HuggingFace Router via `openai` Python SDK (using `gpt-oss-120b:novita`)

---

## 🛠️ Prerequisites & System Requirements

Because this app utilizes native image parsing and OCR for scanned documents, you must have the following system dependencies installed on your Windows machine:

1. **Python 3.9+**
2. **Tesseract OCR:** 
   - Download and install Tesseract-OCR for Windows.
   - *Note: The code expects it at `C:\Program Files\Tesseract-OCR\tesseract.exe`. If you install it elsewhere, update `pytesseract.pytesseract.tesseract_cmd` in `pdf.py`.*
3. **Poppler (for `pdf2image`):**
   - Download Poppler for Windows.
   - *Note: The code expects it at `C:\Users\tikuj\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin`. You **must** update this path in `pdf.py` (Line 147) mapped to your local installation directory for OCR to function.*

---

## 💻 Installation

**1. Clone the Repository:**
```bash
git clone <repository_url>
cd PDFCHAT
```

**2. Create and Activate a Virtual Environment:**
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies:**
Ensure your virtual environment is active, then run:
```bash
pip install -r requirements.txt
```
*(Note: Streamlit might crash if it attempts to lazily crawl local libraries. We have configured `.streamlit/config.toml` to disable the file watcher. Do not delete this file!)*

---

## 🔑 Configuration

### API Keys
The application uses HuggingFace's Router to talk to LLMs natively through the OpenAI Python package.
By default, the AI interacts with a hardcoded `HF_API_KEY` in `app.py`. 
For deploying or sharing, **export your own HuggingFace API Token** to your operating system's environment variables:
```bash
# Windows Command Prompt
set HF_API_KEY="your_huggingface_token_here"

# Windows PowerShell
$env:HF_API_KEY="your_huggingface_token_here"
```

---

## 🏁 How to Start (Usage Guide)

Follow these exact steps from your terminal to boot up the application:

1. **Open your Terminal/Command Prompt.**
2. **Activate the environment** (if not already active):
   ```powershell
   .\venv\Scripts\activate
   ```
3. **Run the Streamlit application:**
   ```powershell
   streamlit run app.py
   ```
4. **Open your Browser:** Streamlit will start a local web server, usually at `http://localhost:8501`. A browser window should open automatically.

### Using the App:
1. **Upload:** Look at the left sidebar. Select **Single** or **Multiple** documents format and drag & drop your files.
2. **Process:** Click the **⚡ Process** button. Wait a few moments as the backend chunks texts and vectorizes paragraphs into FAISS. 
3. **Chat:** Ask questions from the prompt at the bottom of the screen. The AI will cite pages and source documents when formulating its response!
4. **Export:** Satisfied with the session? Download your Q&A history as a JSON file using the **📥 Download JSON** button.

---

## 📂 Project Structure

```text
📂 PDFCHAT
 ┣ 📜 app.py              # Main frontend Streamlit application & Chat UI
 ┣ 📜 pdf.py             # Backend engine: Model loading, PyMuPDF, FAISS, and LLM Logic
 ┣ 📜 requirements.txt    # Python dependencies
 ┣ 📂 .streamlit/         # Configuration locking Streamlit's file-watcher logic
 ┣ 📂 doc_cache/          # Auto-generated directory for storing raw uploaded PDFs
 ┗ 📂 cache/              # Auto-generated directory containing FAISS DBs and serialized objects
```

## 🤝 Troubleshooting & Missing Modules

- **"ModuleNotFoundError":** If terminal throws an missing module error (like `torchvision`), ensure your virtual environment is explicitly activated before hitting `streamlit run`. 
- **OCR errors:** Ensure `poppler_path` inside `pdf.py` points perfectly to your system's `poppler/bin` directory!