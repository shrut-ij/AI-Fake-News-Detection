# AI Fake News Detection (Groq + Brave Search + Streamlit)

An interactive **AI-powered fake news detection tool** built with **Streamlit**, **Groq-hosted LLaMA 3.3 70B**, and **Brave Search API**.  
The app takes a user statement, searches for relevant fact-checking evidence, and provides a **verdict** (TRUE, FALSE, PARTIALLY TRUE, or UNVERIFIABLE) along with explanations, context, and references.

---

## Features

- **Web search integration** using Brave Search API to gather supporting evidence.
- **Real-time AI analysis** using Groq-hosted **LLaMA 3.3 70B** for fact-checking.
- **Structured results** with verdict, explanation, context, and source references.
- **Interactive Streamlit UI** for a smooth user experience.
- **Streaming responses** for faster feedback while the model is thinking.

---

## 🛠Tech Stack

- **[Streamlit](https://streamlit.io/)** – Web app framework.
- **[Groq API](https://groq.com/)** – Access to LLaMA 3.3 70B for analysis.
- **[Brave Search API](https://brave.com/search/api/)** – Evidence gathering via search.
- **[Python](https://www.python.org/)** – Backend logic and async workflows.
- **[httpx](https://www.python-httpx.org/)** – Async HTTP requests.

---

## App URL: https://ai-fake-news-detection.streamlit.app/
