# AI Note Assistant

Local-first notes with **semantic search** and **RAG Q&A** over your own content.
- SQLite for storage
- ChromaDB + `all-MiniLM-L6-v2` embeddings
- Optional LLM synthesis (set `OPENAI_API_KEY`)
- Import/Export (CSV/JSON)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
