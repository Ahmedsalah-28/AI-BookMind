import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import (
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    CombinedMemory
)

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
    buffer_memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=False,
        k=5  # 🆕 keep only last 5 turns
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
       "Your answer (clear, concise, and factual):\n"
       "If the user asks you to simplify or explain in another way, "
       "only simplify the same concept from the context, without introducing new unrelated topics."

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
        # 🆕 UI-friendly error message
        st.error("⚠️ The model request failed. Please try again later.")
        # 🆕 Save to error log
        with open("error_log.txt", "a", encoding="utf-8") as f:
            f.write(str(e) + "\n")
        return "⚠️ The model request failed. Please try again later."
    
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



def rewrite_question(user_q, memory):
    mem_vars = memory.load_memory_variables({})
    chat_history = mem_vars.get("chat_history", "")
    history = mem_vars.get("history", "")
    
    prompt = (
        f"Conversation so far:\n{history}\n\n"
        f"Recent turns:\n{chat_history}\n\n"
        f"User's latest question: {user_q}\n\n"
        "Rewrite the question so it is standalone and fully clear. "
        "Only return the rewritten question."
    )
    resp = ollama_llm.invoke(prompt)
    return str(resp).strip()


def run_rag(user_q: str):
    with st.spinner("Retrieving context..."):
        faiss_query = rewrite_question(user_q, memory)
        results_with_scores = vectorstore.similarity_search_with_score(faiss_query, k=TOP_K_RESULTS)
        filtered = [doc for doc, score in results_with_scores if score > 0.4]
        if not filtered:
            filtered = [doc for doc, score in results_with_scores]
        sources = format_sources(filtered)

    final_prompt = build_final_prompt(user_q, filtered)
    with st.spinner("Generating answer..."):
        answer = get_llm_response(final_prompt) or "⚠️ No answer generated."
    # memory.save_context({"question": user_q}, {"answer": answer})
    return answer, sources



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
    last_user = None
    for m in reversed(st.session_state.messages):
        if m["role"] == "user":
            last_user = m["content"]
            break
    if last_user:
        with st.spinner("Re-generating..."):
            new_answer, sources = run_rag(last_user)

            # Replace last assistant message or append if missing
            replaced = False
            for i in range(len(st.session_state.messages) - 1, -1, -1):
                if st.session_state.messages[i]["role"] == "assistant":
                    st.session_state.messages[i]["content"] = new_answer
                    replaced = True
                    break
            if not replaced:
                st.session_state.messages.append(
                    {"role": "assistant", "content": new_answer}
                )

            # Save to memory
            memory.save_context({"question": last_user}, {"answer": new_answer})
            print(sources)
            # UI: show updated assistant message
            with st.chat_message("assistant"):
                st.markdown(new_answer)

                # Expander for sources
                if sources:
                    with st.expander("📖 Sources"):
                        for s in sources:
                            st.markdown(f"- {s}")


# ==========================
# Chat input flow
# ==========================
user_q = st.chat_input("Ask about AI, Machine Learning, or LLMs...")
if user_q:
    # UI: show user
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)
        
    answer,sources=run_rag(user_q)
    if answer:
        # UI: show assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

            # Expander for sources
            if sources:
                with st.expander("📖 Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

        # Save pair to memory
        memory.save_context({"question": user_q}, {"answer": answer})
