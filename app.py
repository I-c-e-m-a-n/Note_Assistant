import os
import re
import time
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
import pandas as pd

# Embeddings + Vector DB
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Rich text editing
from streamlit_quill import st_quill
from bs4 import BeautifulSoup
import pathlib
import hashlib



# Optional LLM synthesis
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# -----------------------------
# Paths & constants
# -----------------------------
DB_PATH = os.getenv("AINS_DB_PATH", "notes.db")
CHROMA_PATH = os.getenv("AINS_CHROMA_PATH", "data/chroma")
EMB_NAME = os.getenv("AINS_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("AINS_TOP_K", "5"))
DB_PATH = os.getenv("AINS_DB_PATH", "notes.db")
DEFAULT_TOP_K = 5

os.makedirs("data", exist_ok=True)

# -----------------------------
# SQLite helpers
# -----------------------------
def _init_db():
    with closing(sqlite3.connect(DB_PATH)) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            tags TEXT,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """)
        con.commit()


def get_db():
    """
    Return a SQLite3 connection with row_factory set to sqlite3.Row
    so you can access columns as dict keys.
    """
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def insert_note(title: str, tags: str, content: str) -> int:
    now = datetime.utcnow().isoformat(timespec="seconds")
    with closing(sqlite3.connect(DB_PATH)) as con:
        cur = con.cursor()
        cur.execute("INSERT INTO notes (title, tags, content, created_at, updated_at) VALUES (?,?,?,?,?)",
                    (title, tags, content, now, now))
        con.commit()
        return cur.lastrowid

def update_note(note_id: int, title: str, tags: str, content: str):
    now = datetime.utcnow().isoformat(timespec="seconds")
    with closing(sqlite3.connect(DB_PATH)) as con:
        con.execute("UPDATE notes SET title=?, tags=?, content=?, updated_at=? WHERE id=?",
                    (title, tags, content, now, note_id))
        con.commit()

def delete_note(note_id: int):
    with closing(sqlite3.connect(DB_PATH)) as con:
        con.execute("DELETE FROM notes WHERE id=?", (note_id,))
        con.commit()

def fetch_notes() -> pd.DataFrame:
    with closing(sqlite3.connect(DB_PATH)) as con:
        df = pd.read_sql_query("SELECT * FROM notes ORDER BY updated_at DESC", con)
    return df

def keyword_search_df(q: str) -> pd.DataFrame:
    like = f"%{q}%"
    with closing(sqlite3.connect(DB_PATH)) as con:
        df = pd.read_sql_query(
            "SELECT * FROM notes WHERE title LIKE ? OR tags LIKE ? OR content LIKE ? ORDER BY updated_at DESC",
            con, params=(like, like, like))
    return df


# -----------------------------
# Embeddings + Chroma helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    model = SentenceTransformer(EMB_NAME)
    return model

@st.cache_resource(show_spinner=False)
def get_chroma():
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection("notes")
    return client, col

def upsert_embeddings(arg, title=None, content_html=None, tags="", updated_at=""):
    """
    Backward-compatible upsert for Chroma embeddings.

    Accepts EITHER:
      1) A dict: {"id", "title", "content", "tags", "updated_at"}  (current call sites)
      2) Positional args: (id, title, content_html, tags, updated_at)  (older call sites)

    Stores HTML in SQLite, but embeds only clean text for better semantic search.
    """
    # Normalize inputs
    if isinstance(arg, dict):
        note_id     = int(arg["id"])
        title_v     = arg.get("title", "")
        content_v   = arg.get("content", "")        # HTML stored in DB
        tags_v      = arg.get("tags", "") or ""
        updated_v   = arg.get("updated_at", "")
    else:
        # positional/keyword style
        note_id     = int(arg)
        title_v     = title or ""
        content_v   = content_html or ""
        tags_v      = tags or ""
        updated_v   = updated_at or ""

    # Convert HTML -> plain text for embeddings
    text_doc = f"{title_v}\n{html_to_text(content_v)}"

    # Upsert into Chroma
    _, col = get_chroma()
    emb = get_embedder().encode([text_doc], convert_to_numpy=True)[0]
    col.upsert(
        ids=[str(note_id)],
        documents=[text_doc],
        metadatas=[{"title": title_v, "tags": tags_v, "updated_at": updated_v}],
        embeddings=[emb.tolist()],
    )

def delete_embedding(note_id: int):
    _, col = get_chroma()
    try:
        col.delete(ids=[str(note_id)])
    except Exception:
        pass

def semantic_search(q: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    _, col = get_chroma()
    query_emb = get_embedder().encode([q], convert_to_numpy=True)[0].tolist()
    res = col.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["metadatas", "distances", "documents"]  # <- removed "ids"
    )
    out = []
    if res and res.get("ids") and len(res["ids"]) > 0 and len(res["ids"][0]) > 0:
        for i in range(len(res["ids"][0])):
            out.append({
                "id": int(res["ids"][0][i]),  # still available by default
                "distance": float(res["distances"][0][i]),
                "title": res["metadatas"][0][i].get("title", ""),
                "tags": res["metadatas"][0][i].get("tags", ""),
                "updated_at": res["metadatas"][0][i].get("updated_at", ""),
                "document": res["documents"][0][i],
            })
    return out


# -----------------------------
# Optional LLM synthesis (RAG)
# -----------------------------
def llm_answer(question: str, passages: List[str]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if OPENAI_AVAILABLE and api_key:
        client = OpenAI(api_key=api_key)
        system = "You are a helpful assistant that answers strictly using the provided notes. If unsure, say you don't have enough information."
        context = "\n\n".join(f"- {p}" for p in passages)
        prompt = f"Answer the question using only these notes:\n{context}\n\nQuestion: {question}\nAnswer:"
        resp = client.chat.completions.create(
            model=os.getenv("AINS_OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()
    else:
        # Extractive fallback: return top snippets concatenated as a “summary”
        joined = "\n\n".join(passages[:3])
        return f"(Extractive answer; set OPENAI_API_KEY for synthesis)\n\n{joined}"

# -----------------------------
# Utilities
# -----------------------------
def clean_tags(tags: str) -> str:
    if not tags:
        return ""
    parts = [re.sub(r"\s+", "", p) for p in tags.split(",")]
    parts = [p for p in parts if p]
    return ",".join(sorted(set(parts)))

def export_notes(fmt: str) -> bytes:
    df = fetch_notes()
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8")
    elif fmt == "json":
        return df.to_json(orient="records", indent=2).encode("utf-8")
    else:
        raise ValueError("Unsupported export format")

def import_notes(file, fmt: str) -> int:
    if fmt == "csv":
        df = pd.read_csv(file)
    elif fmt == "json":
        df = pd.read_json(file)
    else:
        raise ValueError("Unsupported import format")
    required = {"title", "content"}
    if not required.issubset(df.columns):
        raise ValueError("Import requires at least columns: title, content (optional: tags)")
    count = 0
    for _, r in df.iterrows():
        nid = insert_note(r.get("title", "Untitled"), clean_tags(str(r.get("tags", ""))), str(r.get("content", "")))
        row = {"id": nid, "title": r.get("title", "Untitled"), "tags": clean_tags(str(r.get("tags", ""))),
               "content": str(r.get("content", "")), "updated_at": datetime.utcnow().isoformat(timespec="seconds")}
        upsert_embeddings(row)
        count += 1
    return count

# Store uploaded images
MEDIA_DIR = pathlib.Path(os.getenv("AINS_MEDIA_DIR", "data/media"))
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

def html_to_text(html: str) -> str:
    """Strip HTML to plain text for embeddings."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")

def _insert_image_and_get_snippet(upload_file) -> str:
    """Save uploaded image to MEDIA_DIR and return an <img> HTML snippet."""
    suffix = pathlib.Path(upload_file.name).suffix.lower()
    safe = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{hashlib.blake2b(upload_file.name.encode(), digest_size=4).hexdigest()}{suffix}"
    target = MEDIA_DIR / safe
    with open(target, "wb") as f:
        f.write(upload_file.read())
    return f'<p><img src="{target.as_posix()}" alt="image" style="max-width:100%"></p>'

def _save_image_and_snippet(upload_file):
    """
    Save an uploaded image to MEDIA_DIR and return an HTML <img> snippet
    that references the saved file. Designed to be used with Streamlit's
    st.file_uploader(...).
    """
    # Build a safe, unique filename
    suffix = pathlib.Path(upload_file.name).suffix.lower() or ".png"
    uid = hashlib.blake2b(upload_file.name.encode("utf-8"), digest_size=4).hexdigest()
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    fname = f"{ts}_{uid}{suffix}"
    target = MEDIA_DIR / fname

    # Write the file bytes once (read() exhausts the stream)
    data = upload_file.read()
    with open(target, "wb") as f:
        f.write(data)

    # Return an HTML snippet that will render in the preview/editor
    return f'<p><img src="{target.as_posix()}" alt="{fname}" style="max-width:100%"></p>'

def _build_table_html(rows: int, cols: int, header: bool) -> str:
    parts = ["<table style='border-collapse:collapse;width:100%'>"]
    for r in range(rows):
        parts.append("<tr>")
        for c in range(cols):
            tag = "th" if header and r == 0 else "td"
            parts.append(
                f"<{tag} style='border:1px solid #ddd;padding:6px'>"
                f"{'Header' if tag=='th' else 'Cell'} {r+1}:{c+1}</{tag}>"
            )
        parts.append("</tr>")
    parts.append("</table>")
    return "\n".join(parts)

def rich_editor(key: str, initial_html: str = "") -> str:
    """
    Render a Quill editor with a small toolbox (image upload + table builder) and a live preview.
    Returns the HTML string of the editor content.
    """
    # keep editor state in session so we can append snippets
    state_key = f"{key}_html"
    if state_key not in st.session_state:
        st.session_state[state_key] = initial_html or ""

    left, right = st.columns([3, 2])

    with left:
        st.markdown("Content (Rich Text)")
        toolbar = [
            [{"header": [1, 2, 3, 4, False]}],
            ["bold", "italic", "underline", "strike"],
            [{"list": "ordered"}, {"list": "bullet"}],
            [{"align": []}],
            ["blockquote", "code-block"],
            ["link", "image"],
            ["clean"],
        ]
        html_value = st_quill(
            value=st.session_state[state_key],
            placeholder="Write your note…",
            html=True, toolbar=toolbar, key=f"quill_{key}",
        )

        # Image upload → save to disk → insert <img>
        st.markdown("Insert Image")
        img = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "gif"], key=f"uploader_{key}")
        if img is not None:
            snippet = _insert_image_and_get_snippet(img)
            html_value = (html_value or "") + "\n" + snippet
            st.session_state[state_key] = html_value
            st.success("Image inserted.")
            st.rerun()

        with st.expander("Insert Table"):
            c1, c2 = st.columns(2)
            rows = c1.number_input("Rows", min_value=1, max_value=20, value=2, step=1, key=f"rows_{key}")
            cols = c2.number_input("Columns", min_value=1, max_value=10, value=3, step=1, key=f"cols_{key}")
            is_header = st.checkbox("First row is header", value=True, key=f"hdr_{key}")
            if st.button("Insert Table", key=f"btn_table_{key}"):
                t = _build_table_html(int(rows), int(cols), bool(is_header))
                html_value = (html_value or "") + "\n" + t
                st.session_state[state_key] = html_value
                st.success("Table inserted.")
                st.rerun()

    with right:
        st.markdown("Preview")
        st.markdown(st.session_state[state_key], unsafe_allow_html=True)

    # sync return value from the widget back into session
    st.session_state[state_key] = html_value or st.session_state[state_key]
    return st.session_state[state_key]

def fetch_note_by_id(note_id: int):
    """Return one note row (dict-like) with full HTML content."""
    con = get_db()
    row = con.execute(
        "SELECT id, title, tags, content, updated_at FROM notes WHERE id=?",
        (note_id,)
    ).fetchone()
    return row


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Note Assistant", page_icon="✍️", layout="wide")
st.title("Note Assistant")
st.caption("Semantic notes • Keyword search • RAG Q&A • CSV/JSON import/export  |  Local by default, LLM optional")

with st.sidebar:
    st.header("Settings")
    st.write("**Storage**")
    st.code(f"SQLite: {DB_PATH}\nChroma:  {CHROMA_PATH}", language="text")
    st.write("**Models**")
    st.code(f"Embedding: {EMB_NAME}\nTop-K:    {TOP_K}", language="text")
    if OPENAI_AVAILABLE:
        st.write("**LLM**")
        st.code(f"OPENAI_API_KEY={'set' if os.getenv('OPENAI_API_KEY') else 'not set'}", language="text")
    st.divider()
    st.write("**Import / Export**")
    exp = st.selectbox("Export format", ["csv", "json"], index=0)
    if st.button("Export"):
        data = export_notes(exp)
        st.download_button("Download", data=data, file_name=f"notes.{exp}", mime="text/plain")
    up = st.file_uploader("Import CSV/JSON", type=["csv", "json"])
    if up is not None:
        fmt = "csv" if up.name.lower().endswith(".csv") else "json"
        try:
            added = import_notes(up, fmt)
            st.success(f"Imported {added} notes.")
        except Exception as e:
            st.error(f"Import failed: {e}")

# Ensure DB / Vector store exist
_init_db()
get_embedder()
get_chroma()

tab_new, tab_search, tab_qna, tab_all = st.tabs(["➕ New / Edit", "🔎 Search", "❓ Ask (RAG)", "📚 All Notes"])

# -----------------------------
# New/Edit Note tab
# -----------------------------
with tab_new:
    st.subheader("Create / Update Note")
    df = fetch_notes()
    mode = st.radio("Mode", ["Create", "Edit"], horizontal=True)

    # Rich editor toolbar
    quill_toolbar = [
        [{"header": [1, 2, 3, 4, False]}],
        ["bold", "italic", "underline", "strike"],
        [{"list": "ordered"}, {"list": "bullet"}],
        [{"align": []}],
        ["blockquote", "code-block"],
        ["link", "image"],
        ["clean"],
    ]

    if mode == "Edit" and not df.empty:
        options = {f"[{r['id']}] {r['title']}": int(r["id"]) for _, r in df.iterrows()}
        sel = st.selectbox("Select a note to edit", list(options.keys()))
        note_id = options[sel]
        row = df[df["id"] == note_id].iloc[0]

        title = st.text_input("Title", value=row["title"])
        tags = st.text_input("Tags (comma-separated)", value=row.get("tags") or "")

        # Rich editor + Preview
        left, right = st.columns([3, 2], vertical_alignment="top")
        with left:
            st.markdown("Content (Rich Text)")
            editor_html = st_quill(
                value=row["content"] or "",     # store/render HTML in DB
                placeholder="Write your note…",
                html=True,
                toolbar=quill_toolbar,
                key=f"quill_edit_{note_id}",
            )

            # Image upload → save file → insert <img>
            st.markdown("Insert Image")
            img = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "gif"], key=f"uploader_edit_{note_id}")
            if img is not None:
                snippet = _save_image_and_snippet(img)
                editor_html = (editor_html or "") + "\n" + snippet
                # force update of the quill value by re-rendering with new default:
                # Streamlit cannot set the internal state of the widget directly,
                # so we persist by saving on "Save" below.

            with st.expander("Insert Table"):
                c1, c2 = st.columns(2)
                rows_num = c1.number_input("Rows", min_value=1, max_value=20, value=2, step=1, key=f"rows_edit_{note_id}")
                cols_num = c2.number_input("Columns", min_value=1, max_value=10, value=3, step=1, key=f"cols_edit_{note_id}")
                has_header = st.checkbox("First row is header", value=True, key=f"hdr_edit_{note_id}")
                if st.button("Insert Table", key=f"btn_table_edit_{note_id}"):
                    t_html = _build_table_html(int(rows_num), int(cols_num), bool(has_header))
                    editor_html = (editor_html or "") + "\n" + t_html

        with right:
            st.markdown("Preview")
            st.markdown(editor_html or "", unsafe_allow_html=True)

        cols = st.columns(3)
        if cols[0].button("💾 Save", key=f"save_{note_id}"):
            update_note(note_id, title, clean_tags(tags), editor_html or "")
            upsert_embeddings({
                "id": note_id,
                "title": title,
                "tags": clean_tags(tags),
                "content": editor_html or "",
                "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
            })
            st.success("Note updated.")
            st.rerun()

        if cols[1].button("🗑️ Delete", key=f"del_{note_id}"):
            delete_note(note_id)
            delete_embedding(note_id)
            st.success("Note deleted.")
            st.rerun()

    else:
        title = st.text_input("Title")
        tags = st.text_input("Tags (comma-separated)")

        # Rich editor + Preview for new note
        left, right = st.columns([3, 2], vertical_alignment="top")
        with left:
            st.markdown("Content (Rich Text)")
            editor_html = st_quill(
                value="",
                placeholder="Write your note…",
                html=True,
                toolbar=quill_toolbar,
                key="quill_create",
            )

            st.markdown("Insert Image")
            img = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "gif"], key="uploader_create")
            if img is not None:
                snippet = _save_image_and_snippet(img)
                editor_html = (editor_html or "") + "\n" + snippet

            with st.expander("Insert Table"):
                c1, c2 = st.columns(2)
                rows_num = c1.number_input("Rows", min_value=1, max_value=20, value=2, step=1, key="rows_create")
                cols_num = c2.number_input("Columns", min_value=1, max_value=10, value=3, step=1, key="cols_create")
                has_header = st.checkbox("First row is header", value=True, key="hdr_create")
                if st.button("Insert Table", key="btn_table_create"):
                    t_html = _build_table_html(int(rows_num), int(cols_num), bool(has_header))
                    editor_html = (editor_html or "") + "\n" + t_html

        with right:
            st.markdown("Preview")
            st.markdown(editor_html or "", unsafe_allow_html=True)

        if st.button("➕ Add Note", key="add_note"):
            if not title.strip() or not (editor_html or "").strip():
                st.error("Title and Content are required.")
            else:
                nid = insert_note(title.strip(), clean_tags(tags), (editor_html or "").strip())
                upsert_embeddings({
                    "id": nid,
                    "title": title.strip(),
                    "tags": clean_tags(tags),
                    "content": (editor_html or "").strip(),
                    "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
                })
                st.success(f"Note added with id {nid}.")
                st.rerun()


# -----------------------------
# Search tab  (preview above results)
# -----------------------------
with tab_search:
    st.subheader("Search Notes")
    q = st.text_input("Query", placeholder="e.g., gradient descent; invoice tax 2024; meeting notes")

    c1, c2, c3 = st.columns(3)
    use_keyword = c1.checkbox("Keyword", value=True)
    use_semantic = c2.checkbox("Semantic", value=True)
    k = c3.slider("Top-K (semantic)", 1, 15, DEFAULT_TOP_K)

    # State for preview
    st.session_state.setdefault("preview_note_id", None)
    st.session_state.setdefault("preview_note_title", "")
    st.session_state.setdefault("preview_note_meta", "")
    st.session_state.setdefault("preview_note_html", "")

    # Run search
    if st.button("Search"):
        if not q.strip():
            st.warning("Enter a query.")
        else:
            st.session_state["search_started"] = True
            st.session_state["search_query"] = q.strip()

    # PREVIEW FIRST (always on top)
    st.markdown("### Preview")
    if st.session_state.get("preview_note_id") is None:
        st.caption("Select a result below to preview its full content.")
    else:
        st.markdown(f"**{st.session_state.get('preview_note_title','')}**")
        st.caption(st.session_state.get("preview_note_meta",""))
        st.markdown(st.session_state.get("preview_note_html",""), unsafe_allow_html=True)

    st.divider()

    # RESULTS BELOW PREVIEW
    if st.session_state.get("search_started"):
        q_run = st.session_state.get("search_query", "")

        # Keyword matches
        if use_keyword:
            st.markdown("### Keyword matches")
            try:
                df_kw = keyword_search_df(q_run)
            except Exception as e:
                st.error(f"Keyword search failed: {e}")
                df_kw = None

            if df_kw is not None and not df_kw.empty:
                for _, r in df_kw.iterrows():
                    nid = int(r["id"])
                    title = r["title"]
                    meta = f"#{nid} • {(r.get('tags','') or '')} • {r['updated_at']}"
                    row_c1, row_c2 = st.columns([6, 1])
                    with row_c1:
                        st.caption(meta)
                        if st.button(title, key=f"kw_title_{nid}"):
                            note = fetch_note_by_id(nid)
                            if note:
                                st.session_state["preview_note_id"] = nid
                                st.session_state["preview_note_title"] = note["title"]
                                st.session_state["preview_note_meta"] = f"Tags: {note['tags'] or ''} • Updated: {note['updated_at']}"
                                st.session_state["preview_note_html"] = note["content"] or ""
                                st.rerun()
                    with row_c2:
                        if st.button("Preview", key=f"kw_prev_{nid}"):
                            note = fetch_note_by_id(nid)
                            if note:
                                st.session_state["preview_note_id"] = nid
                                st.session_state["preview_note_title"] = note["title"]
                                st.session_state["preview_note_meta"] = f"Tags: {note['tags'] or ''} • Updated: {note['updated_at']}"
                                st.session_state["preview_note_html"] = note["content"] or ""
                                st.rerun()
            else:
                st.caption("No keyword matches.")

            st.divider()

        # Semantic matches
        if use_semantic:
            st.markdown("### Semantic matches")
            try:
                res = semantic_search(q_run, k)
            except Exception as e:
                st.error(f"Semantic search failed: {e}")
                res = []

            if res:
                for r in res:
                    nid = int(r["id"])
                    title = r["title"] or f"Note {nid}"
                    meta = f"#{nid} • {(r.get('tags','') or '')} • dist={r['distance']:.3f}"
                    row_c1, row_c2 = st.columns([6, 1])
                    with row_c1:
                        st.caption(meta)
                        if st.button(title, key=f"sem_title_{nid}"):
                            note = fetch_note_by_id(nid)
                            if note:
                                st.session_state["preview_note_id"] = nid
                                st.session_state["preview_note_title"] = note["title"]
                                st.session_state["preview_note_meta"] = f"Tags: {note['tags'] or ''} • Updated: {note['updated_at']}"
                                st.session_state["preview_note_html"] = note["content"] or ""
                                st.rerun()
                    with row_c2:
                        if st.button("Preview", key=f"sem_prev_{nid}"):
                            note = fetch_note_by_id(nid)
                            if note:
                                st.session_state["preview_note_id"] = nid
                                st.session_state["preview_note_title"] = note["title"]
                                st.session_state["preview_note_meta"] = f"Tags: {note['tags'] or ''} • Updated: {note['updated_at']}"
                                st.session_state["preview_note_html"] = note["content"] or ""
                                st.rerun()
            else:
                st.caption("No semantic matches.")


# -----------------------------
# Q&A tab
# -----------------------------
with tab_qna:
    st.subheader("Ask Questions (RAG)")
    question = st.text_input("Your question")
    k = st.slider("Retrieve Top-K notes", 1, 10, min(TOP_K, 5))
    if st.button("Answer"):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            res = semantic_search(question.strip(), k)
            if not res:
                st.info("No relevant notes found.")
            else:
                passages = [r["document"] for r in res]
                with st.spinner("Thinking..."):
                    ans = llm_answer(question.strip(), passages)
                st.markdown("**Answer**")
                st.write(ans)
                st.markdown("---")
                st.caption("Context used")
                for r in res:
                    with st.expander(f"#{r['id']} • {r['title']} • {r['tags']}"):
                        st.markdown(r["document"])

# -----------------------------
# All notes tab
# -----------------------------
with tab_all:
    st.subheader("All Notes")
    df = fetch_notes()
    st.dataframe(df, use_container_width=True, hide_index=True)
