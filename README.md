# 📰 Fake News Detector — Streamlit App (LLM + Brave Search + Groq Streaming)

A **real-time fake news detection tool** powered by:
- **Groq-hosted LLaMA 3.3 70B** for fast, high-quality analysis  
- **Brave Search API** as the *external knowledge base* (RAG-like behavior, no local embeddings needed)  
- **Streamlit** for an interactive web interface with live streaming responses

---

## 🚀 Features
- **Instant news verification** via Brave Search API  
- **Streaming LLM responses** (Groq API) for a chatbot-like feel  
- **Automatic reference extraction** from top Brave search results  
- **User-friendly interface** — press **Enter** to search or click the Search button  
- No need to download large datasets or models locally — **everything is cloud-based**  

---

## 🛠Tech Stack

- **[Streamlit](https://streamlit.io/)** – Web app framework.
- **[Groq API](https://groq.com/)** – Access to LLaMA 3.3 70B for analysis.
- **[Brave Search API](https://brave.com/search/api/)** – Evidence gathering via search.
- **[Python](https://www.python.org/)** – Backend logic and async workflows.
- **[httpx](https://www.python-httpx.org/)** – Async HTTP requests.

---

## App URL: https://ai-fake-news-detection.streamlit.app/
