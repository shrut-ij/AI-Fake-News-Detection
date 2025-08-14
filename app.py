import os
import re
import io
import time
import asyncio
import textwrap
from typing import List, Dict, Tuple

import streamlit as st
import httpx
from dotenv import load_dotenv

# Optional ML deps
TRANSFORMERS_AVAILABLE = True
try:
    from transformers import pipeline
except Exception:
    TRANSFORMERS_AVAILABLE = False

# Groq
from groq import Groq

# =========================
# Load environment variables
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
FAKE_NEWS_MODEL = os.getenv(
    "FAKE_NEWS_MODEL", "mrm8488/bert-tiny-finetuned-fake-news-detection"
)

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# =========================
# Utilities
# =========================
def sanitize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
HEADERS_BRAVE = {"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY or ""}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
HTTP_HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/html,application/json"}


async def brave_search(query: str, count: int = 6) -> List[Dict]:
    if not BRAVE_API_KEY:
        return []
    params = {"q": query, "count": count, "freshness": "pd"}
    async with httpx.AsyncClient(timeout=20) as ac:
        r = await ac.get(BRAVE_SEARCH_URL, headers=HEADERS_BRAVE, params=params)
        r.raise_for_status()
        data = r.json()
    results = data.get("web", {}).get("results", [])
    out = []
    for r in results:
        out.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": sanitize_text(r.get("description") or r.get("snippet") or ""),
                "site": r.get("profile", {}).get("name", ""),
            }
        )
    return out


# =========================
# DL Classifier
# =========================
CLASSIFIER = None
if TRANSFORMERS_AVAILABLE:
    try:
        CLASSIFIER = pipeline(
            "text-classification",
            model=FAKE_NEWS_MODEL,
            tokenizer=FAKE_NEWS_MODEL,
            truncation=True,
            top_k=None,
        )
    except Exception:
        CLASSIFIER = None


def run_classifier(text: str) -> Dict:
    if not text or not CLASSIFIER:
        return {"label": "UNKNOWN", "score": 0.0, "raw": None}
    preds = CLASSIFIER(text[:4000])
    label = preds[0]["label"] if isinstance(preds, list) else preds["label"]
    score = preds[0]["score"] if isinstance(preds, list) else preds["score"]
    mapped = "REAL" if "non" in label.lower() or "neutral" in label.lower() else "FAKE"
    return {"label": mapped, "score": float(score), "raw": preds}


# =========================
# Groq LLM Streaming
# =========================
FACTCHECK_SYSTEM = (
    "You are a rigorous fact-checking assistant. "
    "Use only the supplied evidence. Cite sources. "
    "If evidence is insufficient, say UNVERIFIABLE. "
    "Output EXACTLY in this format:\n\n"
    "<verdict>TRUE/FALSE/PARTIALLY TRUE/UNVERIFIABLE</verdict>\n\n"
    "<explanation>\nDetailed reasoning with citations.\n</explanation>\n\n"
    "<references>\n1. Source - URL\n</references>\n"
)


async def stream_factcheck(statement: str, evidence_text: str) -> str:
    if not client:
        yield (
            "<verdict>UNVERIFIABLE</verdict>\n\n"
            "<explanation>Groq API key missing.</explanation>\n\n"
            "<references></references>"
        )
        return

    messages = [
        {"role": "system", "content": FACTCHECK_SYSTEM},
        {"role": "user", "content": f"Statement: {statement}\n\nEvidence:\n{evidence_text}"},
    ]

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
        max_tokens=1200,
        top_p=1,
        stream=True,
    )

    output = ""
    for chunk in resp:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            output += content
            yield output


def extract_sections(text: str) -> Tuple[str, str, str]:
    def get(tag: str) -> str:
        m = re.search(fr"<{tag}>(.*?)</{tag}>", text or "", re.DOTALL | re.IGNORECASE)
        return sanitize_text(m.group(1)) if m else ""

    verdict = get("verdict")
    explanation = get("explanation")
    references = get("references")
    return verdict, explanation, references

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("🔍 AI Fake News Detection")
st.caption("DL classifier + Online RAG (Brave Search) + Groq LLM explanations")

user_input = st.text_input("Enter a news headline or article:", key="news_input")

if user_input:
    async def handle_query(query_str: str):
        # DL classifier
        clf_start = time.time()
        clf_res = run_classifier(query_str)
        clf_time = time.time() - clf_start

        st.markdown("### 🧪 Classifier Result")
        st.write(f"**Prediction:** {clf_res['label']}  ")
        st.write(f"**Confidence:** {clf_res['score']:.2f}")

        # Brave Search
        with st.spinner("🔎 Searching evidence via Brave..."):
            results = await brave_search(f"fact check: {query_str}")
            evidence_lines = []
            for r in results[:6]:
                evidence_lines.append(f"- {r['title']} - {r['url']}\nSnippet: {r['snippet']}")
            evidence_text = "\n".join(evidence_lines)
            st.success(f"Found {len(results)} sources")
            if results:
                st.markdown("### 📎 Evidence")
                for line in evidence_lines:
                    st.markdown(line)

        # LLM Streaming
        with st.spinner("🤖 Analyzing with Groq LLM..."):
            ph = st.empty()
            latest = ""
            async for partial in stream_factcheck(query_str, evidence_text):
                latest = partial
                # ph.markdown(partial)

        verdict, explanation, references = extract_sections(latest)

        st.markdown("---")
        st.markdown(f"### 🧾 Verdict: **{verdict or 'UNVERIFIABLE'}**")
        st.markdown("### 🔎 Explanation")
        st.markdown(explanation or "(no explanation)")
        if references:
            st.markdown("### 📚 References")
            
            # Split references by number pattern
            refs = re.split(r'\s*\d+\.\s*', references)
            refs = [ref.strip() for ref in refs if ref.strip()]  # remove empty entries

            # Format as a numbered markdown list
            formatted_refs = "\n".join([f"{i+1}. {ref}" for i, ref in enumerate(refs)])

            st.markdown(formatted_refs)

    asyncio.run(handle_query(user_input))
