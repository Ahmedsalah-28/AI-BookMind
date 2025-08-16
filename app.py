import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory, CombinedMemory
from langchain_community.llms import Ollama

from embedding_utils import get_embeddings
from config import (
    INDEX_DIR, LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_TOKENS,
    MODEL, API_URL, TOP_K_RESULTS
)

# ==========================
# Environment & Page Setup
# ==========================
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

st.set_page_config(page_title="AI/ML RAG Chat", page_icon="🤖", layout="wide")
st.title("📚 AI BookMind — Talk to Your AI & ML Books Instantly (RAG + Memory)")

# ==========================
# Sidebar Controls
# ==========================
with st.sidebar:
    st.subheader("Controls")
    do_regen = st.button("🔁 Regenerate last answer")
    if st.button("🧹 Clear chat & memory"):
        for k in ["messages", "memory", "ollama_llm"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    st.markdown("---")
    st.caption("OpenRouter model / params (from config.py)")
    st.text(f"Model: {MODEL}")
    st.text(f"Temp:  {LLM_TEMPERATURE}")
    st.text(f"TopP:  {LLM_TOP_P}")
    st.text(f"MaxTk: {LLM_MAX_TOKENS}")

# ==========================
# Load FAISS once
# ==========================
if "vectorstore" not in st.session_state:
    try:
        st.session_state.vectorstore = FAISS.load_local(
            INDEX_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"❌ Failed to load vectorstore: {e}")
        st.stop()

vectorstore = st.session_state.vectorstore

# ==========================
# Init Memory (Ollama)
# ==========================
if "ollama_llm" not in st.session_state:
    st.session_state.ollama_llm = Ollama(model="mistral", temperature=0)

ollama_llm = st.session_state.ollama_llm

if "memory" not in st.session_state:
    summary_memory = ConversationSummaryMemory(
        llm=ollama_llm,
        memory_key="history",  
        input_key="question",
        output_key="answer",
        max_token_limit=1200
    )
    buffer_memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=False
    )
    st.session_state.memory = CombinedMemory(
        memories=[summary_memory, buffer_memory]
    )

memory = st.session_state.memory

# ==========================
# Prompt Template (improved)
# ==========================
prompt_template = PromptTemplate(
    input_variables=["history", "chat_history", "context", "question"],
    template=(
        "You are a highly accurate AI assistant specialized in AI, ML, and LLMs.\n"
        "Always answer ONLY from the provided context. If the answer is not in the context, reply with:\n"
        "\"I don't know based on the provided data.\"\n\n"
        "Conversation summary:\n{history}\n\n"
        "Recent turns (verbatim):\n{chat_history}\n\n"
        "Relevant context:\n{context}\n\n"
        "User question:\n{question}\n\n"
        "Your answer (clear, concise, and factual):"
    )
)

# ==========================
# Helpers
# ==========================
def get_llm_response(prompt: str) -> str:
    """Call OpenRouter for final RAG answer"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "max_tokens": LLM_MAX_TOKENS,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant specialized in AI, Machine Learning, and Large Language Models. "
                    "Only answer based on the provided context. If the context does not contain the answer, "
                    "say 'I don't know based on the provided data.'"
                )
            },
            {"role": "user", "content": prompt}
        ]
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        if "choices" in payload and payload["choices"]:
            return payload["choices"][0]["message"]["content"]
        return "⚠️ No response from the model."
    except requests.exceptions.RequestException as e:
        return f"❌ API request failed: {str(e)}"

def format_sources(docs):
    """Extract a compact list of sources from doc.metadata."""
    srcs = []
    for d in docs:
        title = d.metadata.get("book_title") or d.metadata.get("source") or "Unknown"
        page = d.metadata.get("page")
        if page is not None:
            srcs.append(f"{title} (p.{page})")
        else:
            srcs.append(f"{title}")
    # unique & keep order
    seen = set()
    uniq = []
    for s in srcs:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq

def build_final_prompt(question: str, results):
    # get memory vars
    mem_vars = memory.load_memory_variables({})
    history = mem_vars.get("history", "")
    chat_history = mem_vars.get("chat_history", "")

    # context from vectorstore
    context = "\n\n---\n\n".join(doc.page_content for doc in results)

    return prompt_template.format(
        history=history,
        chat_history=chat_history,
        context=context,
        question=question
    )

# ==========================
# Chat State
# ==========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ==========================
# Regenerate last answer
# ==========================
if do_regen and st.session_state.messages:
    # نعيد آخر سؤال
    last_user = None
    for m in reversed(st.session_state.messages):
        if m["role"] == "user":
            last_user = m["content"]
            break
    if last_user:
        with st.spinner("Re-generating..."):
            # Retrieve context again
            results = vectorstore.similarity_search(last_user, k=TOP_K_RESULTS)
            final_prompt = build_final_prompt(last_user, results)
            new_answer = get_llm_response(final_prompt)

            # Replace last assistant message or append if missing
            for i in range(len(st.session_state.messages) - 1, -1, -1):
                if st.session_state.messages[i]["role"] == "assistant":
                    st.session_state.messages[i]["content"] = new_answer
                    break
            else:
                st.session_state.messages.append({"role": "assistant", "content": new_answer})

            # Save to memory
            memory.save_context({"question": last_user}, {"answer": new_answer})

            st.rerun()

# ==========================
# Chat input flow
# ==========================
user_q = st.chat_input("Ask about AI, Machine Learning, or LLMs...")
if user_q:
    # UI: show user
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Retrieve context
    with st.spinner("Retrieving context..."):
        results = vectorstore.similarity_search(user_q, k=TOP_K_RESULTS)
        sources = format_sources(results)

    # Build prompt with memory
    final_prompt = build_final_prompt(user_q, results)

    # Get answer from OpenRouter
    with st.spinner("Generating answer..."):
        answer = get_llm_response(final_prompt)

    # Append Sources
    if sources:
        answer_with_src = answer + "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
    else:
        answer_with_src = answer

    # UI: show assistant
    st.session_state.messages.append({"role": "assistant", "content": answer_with_src})
    with st.chat_message("assistant"):
        st.markdown(answer_with_src)

    # Save pair to memory (AFTER we have answer to avoid key errors)
    memory.save_context({"question": user_q}, {"answer": answer})
