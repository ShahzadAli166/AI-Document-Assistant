# 📚 AI Document Assistant

A modern Streamlit RAG app for PDF, DOCX, TXT and Markdown documents, with local upload and public/shared Google Drive file/folder loading.

## Features
- Multi-file local upload
- Google Drive file/folder loading
- PDF page metadata
- Overlapping chunking
- Sentence Transformers embeddings
- FAISS semantic search
- Keyword search
- Hybrid retrieval (70% semantic + 30% keyword)
- Groq grounded answers
- Retrieved source cards after every answer
- Session-state reuse so document embeddings are not recreated for every question
- Cached embedding model
- Modern chat-style Streamlit UI

## Important Groq model update
The app uses `openai/gpt-oss-120b` by default. Groq deprecated `llama-3.3-70b-versatile` on August 16, 2026. If Groq changes its supported production models, update `GROQ_MODEL` near the top of `app.py`.

## Setup

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install:
```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_real_groq_api_key"
```

Run:
```bash
streamlit run app.py
```

For Streamlit Community Cloud, add `GROQ_API_KEY` in the app's Secrets settings. Never hardcode the key in `app.py`.

## Pipeline

```text
Documents → extraction → chunks → embeddings → FAISS
                                      ↓
Question → embedding + keyword search → hybrid ranking
                                      ↓
                               top chunks → Groq
                                      ↓
                              grounded answer + sources
```

## Google Drive
The app uses `gdown` without the unsupported `fuzzy` argument. File URLs use `gdown.download()` and folder URLs use `gdown.download_folder()`. Only `.pdf`, `.docx`, `.txt`, and `.md` files are sent into the RAG pipeline.

The Drive file/folder must be accessible through its sharing link. The app does not bypass private Drive permissions.

If you see `No supported PDF, DOCX, TXT or MD files were found`, verify that the shared folder actually contains supported file types and that the link points to the file/folder rather than an inaccessible shortcut.

## Processing optimization
Document chunks, embeddings, FAISS index, document metadata, and processed hashes are kept in `st.session_state`. The Sentence Transformer model is loaded with `st.cache_resource`. Asking another question embeds only the question; document embeddings are not recreated.

## Limitations
Scanned/image-only PDFs may need OCR. Private Google Drive content requires an authenticated Drive integration; a public/shared link alone cannot bypass Google permissions.
