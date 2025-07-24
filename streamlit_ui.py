import streamlit as st
import os
import asyncio
from groq import Groq
from dotenv import load_dotenv
import httpx
import time
import re

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Helper to search the web using Brave Search API
async def search_web(query):
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    params = {"q": query, "count": 5}

    async with httpx.AsyncClient() as async_client:
        response = await async_client.get(url, headers=headers, params=params)
        results = response.json().get("web", {}).get("results", [])
        formatted = "\n".join([f"{r['title']} - {r['url']}" for r in results])
        return formatted

# Extract structured sections from model output
def extract_sections(text):
    def get(tag):
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else ""
    return get("verdict"), get("explanation"), get("context"), get("references")

# Stream response from Groq
async def get_analysis(statement, evidence):
    system_prompt = """You are a fact-checking assistant. Based on the evidence provided, analyze the statement and respond in this exact format:

<verdict>TRUE/FALSE/PARTIALLY TRUE/UNVERIFIABLE</verdict>

<explanation>
Your detailed explanation goes here with reasoning and references.
</explanation>

<context>
Additional helpful context for nuance.
</context>

<references>
1. Source Name - URL
2. Source Name - URL
</references>
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Statement: {statement}\n\nEvidence:\n{evidence}"}
    ]

    response = client.chat.completions.create(
        # model="llama-3-70b-8192",  # Groq's LLaMA 3.3 70B Instruct
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        stream=True
    )

    output = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            output += content
            yield output

# Main app
async def main():
    st.set_page_config(page_title="LLaMA 3 Fake News Detection", layout="centered")
    # st.title("🔍 AI Fact Checker (Groq + LLaMA 3.3 70B)")
    st.title("🔍 AI Fake News Detection")
    # st.caption("Backed by Brave search and Groq-hosted LLaMA 3")

    user_input = st.chat_input("Enter a statement to fact-check...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        # Web Search
        with st.chat_message("assistant"):
            with st.spinner("🔎 Searching for evidence..."):
                search_start = time.time()
                evidence = await search_web(f"fact check: {user_input}")
                search_time = time.time() - search_start
                st.success(f"Search completed in {search_time:.2f}s")

        # LLM Analysis
        with st.chat_message("assistant"):
            with st.spinner("🤖 Analyzing..."):
                analysis_start = time.time()
                response_placeholder = st.empty()
                async for partial_response in get_analysis(user_input, evidence):
                    response_placeholder.markdown(partial_response)
                analysis_time = time.time() - analysis_start

        # Final output formatting
        verdict, explanation, context, references = extract_sections(partial_response)
        st.markdown("---")
        st.markdown(f"⏱️ _Search: {search_time:.2f}s, Analysis: {analysis_time:.2f}s_")

        st.markdown(f"### 🧾 Verdict\n**{verdict}**")
        st.markdown("### 🔎 Explanation")
        st.markdown(explanation)
        if context:
            st.markdown("### 🧠 Context")
            st.markdown(context)
        if references:
            st.markdown("### 📚 References")
            st.markdown(references)

if __name__ == "__main__":
    asyncio.run(main())