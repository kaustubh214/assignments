"""
rag.py — 100% Free RAG Pipeline
────────────────────────────────
Embeddings : sentence-transformers/all-MiniLM-L6-v2   (HuggingFace, free, runs locally)
LLM        : HuggingFace Inference API — google/flan-t5-large  (free tier, no credit card)
             OR Ollama locally (llama3 / mistral) if installed
Vector DB  : ChromaDB  (in-memory, free)
PDF        : PyMuPDF   (free)
"""

import os
import re
import requests
from typing import Optional

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ── Constants ──────────────────────────────────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"   # ~90 MB, runs on CPU
HF_API_URL    = "https://api-inference.huggingface.co/models/google/flan-t5-large"
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "mistral"   # change to llama3, phi3, etc.


class RAGPipeline:
    """
    End-to-end RAG pipeline — zero paid APIs.

    LLM priority:
      1. Ollama  (if running locally — best quality)
      2. HuggingFace Inference API  (free tier, needs HF_TOKEN env var)
      3. Extractive fallback  (no token needed — finds best matching chunk)
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 60,
        hf_token: Optional[str] = None,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.hf_token      = hf_token or os.environ.get("HF_TOKEN", "")
        self.vectorstore   = None

        # Load embedding model once (cached by sentence-transformers)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # ────────────────────────────────────────────────────────────────────────
    # STEP 1 — Extract text from PDF
    # ────────────────────────────────────────────────────────────────────────
    def _extract_text(self, file_obj) -> tuple[list[str], int]:
        data = file_obj.read() if hasattr(file_obj, "read") else file_obj
        doc  = fitz.open(stream=data, filetype="pdf")
        pages = []
        for page in doc:
            text = page.get_text("text")
            # Basic cleanup
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]+", " ", text)
            pages.append(text.strip())
        doc.close()
        return pages, len(pages)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 2 — Chunk
    # ────────────────────────────────────────────────────────────────────────
    def _chunk_pages(self, pages: list[str]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        docs = []
        for page_num, text in enumerate(pages, 1):
            if not text.strip():
                continue
            for chunk in splitter.split_text(text):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"page": page_num},
                ))
        return docs

    # ────────────────────────────────────────────────────────────────────────
    # STEP 3+4 — Embed + store in ChromaDB
    # ────────────────────────────────────────────────────────────────────────
    def _build_vectorstore(self, docs: list[Document]):
        return Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name="pdf_rag_free",
        )

    # ────────────────────────────────────────────────────────────────────────
    # STEP 5 — Retrieve top-k chunks
    # ────────────────────────────────────────────────────────────────────────
    def _retrieve(self, question: str, top_k: int = 4) -> list[Document]:
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )
        return retriever.invoke(question)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 6 — Generate answer (three free backends)
    # ────────────────────────────────────────────────────────────────────────

    def _prompt(self, question: str, context: str) -> str:
        return (
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say 'Not found in document'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    def _try_ollama(self, question: str, context: str) -> Optional[str]:
        """Use local Ollama if it's running."""
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": self._prompt(question, context),
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 300},
                },
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception:
            pass
        return None

    def _try_hf_api(self, question: str, context: str) -> Optional[str]:
        """Use HuggingFace Inference API (free tier, needs HF_TOKEN)."""
        if not self.hf_token:
            return None
        try:
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {
                "inputs": self._prompt(question, context),
                "parameters": {
                    "max_new_tokens": 250,
                    "temperature": 0.1,
                    "do_sample": False,
                },
            }
            resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                # flan-t5 returns list of dicts
                if isinstance(data, list) and data:
                    text = data[0].get("generated_text", "").strip()
                    # Remove the prompt echo if present
                    if "Answer:" in text:
                        text = text.split("Answer:")[-1].strip()
                    return text if text else None
        except Exception:
            pass
        return None

    def _extractive_fallback(self, question: str, docs: list[Document]) -> str:
        """
        Zero-dependency fallback: returns the most relevant chunk as the answer.
        No LLM needed — works offline with no tokens.
        """
        q_words = set(question.lower().split())
        best_score = -1
        best_chunk = docs[0].page_content if docs else "No content found."

        for doc in docs:
            words  = set(doc.page_content.lower().split())
            score  = len(q_words & words) / max(len(q_words), 1)
            if score > best_score:
                best_score = best_chunk = None  # reset
                best_score = score
                best_chunk = doc.page_content

        # Trim to a readable length
        sentences = re.split(r"(?<=[.!?])\s+", best_chunk)
        answer    = " ".join(sentences[:6])
        return f"[Extractive answer — most relevant passage]\n\n{answer}"

    def _generate(self, question: str, docs: list[Document]) -> tuple[str, str]:
        """Returns (answer, backend_used)."""
        context = "\n\n---\n\n".join(d.page_content for d in docs)

        # Try Ollama first (best quality, fully local)
        ans = self._try_ollama(question, context)
        if ans:
            return ans, "Ollama (local)"

        # Try HuggingFace Inference API (free tier)
        ans = self._try_hf_api(question, context)
        if ans:
            return ans, "HuggingFace API (free)"

        # Always-available extractive fallback
        ans = self._extractive_fallback(question, docs)
        return ans, "Extractive (offline fallback)"

    # ────────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────────

    def ingest(self, file_obj) -> dict:
        """Process a PDF. Returns stats dict."""
        pages, page_count = self._extract_text(file_obj)
        word_count = sum(len(p.split()) for p in pages)
        docs = self._chunk_pages(pages)
        if not docs:
            raise ValueError(
                "No text extracted. The PDF may be image-only (scanned). "
                "Use an OCR tool to make it text-searchable first."
            )
        self.vectorstore = self._build_vectorstore(docs)
        return {"pages": page_count, "chunks": len(docs), "words": word_count}

    def query(self, question: str, top_k: int = 4) -> dict:
        """
        Ask a question. Returns:
          {"answer": str, "sources": list[str], "backend": str}
        """
        if self.vectorstore is None:
            raise RuntimeError("No document indexed. Call ingest() first.")

        docs    = self._retrieve(question, top_k)
        answer, backend = self._generate(question, docs)

        sources = [
            f"[Page {d.metadata.get('page', '?')}]  {d.page_content[:280]}…"
            for d in docs
        ]
        return {"answer": answer, "sources": sources, "backend": backend}
