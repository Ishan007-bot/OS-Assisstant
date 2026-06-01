"""Open Source Assistant — Streamlit chat UI.
"""
import os
import time

import streamlit as st
from dotenv import load_dotenv

from llm import get_llm
from memory.context import build_context, estimate_tokens
from safety.guardrails import SAFETY_SYSTEM_PROMPT, check_input, filter_output
from observability.logger import log_turn
from tools.toolbox import TOOL_SYSTEM_PROMPT, parse_tool_call, run_tool

load_dotenv()

# System prompt = safety behavior + tool-use instructions.
SYSTEM_PROMPT = SAFETY_SYSTEM_PROMPT + TOOL_SYSTEM_PROMPT
MAX_TOOL_ROUNDS = 2
MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", "2048"))
MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "0")) or None  # 0 -> no hard cap

st.set_page_config(page_title="Open Source Assistant", page_icon="🤖")
st.title("🤖 Open Source Assistant")


@st.cache_resource
def load_backend():
    return get_llm()


llm = load_backend()
st.caption(f"Backend: `{llm.name}`  ·  short-term memory: ~{MAX_TOKENS} tokens")

# --- conversation state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- render existing history ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- handle a new user message ---
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Input guardrail: block clearly harmful requests before the model sees them.
    allowed, refusal = check_input(prompt)
    if not allowed:
        with st.chat_message("assistant"):
            st.markdown(refusal)
        st.session_state.messages.append({"role": "assistant", "content": refusal})
        log_turn(
            backend=llm.name,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=0,
            latency_ms=0,
            blocked=True,
        )
    else:
        # Short-term memory: system prompt + as much recent history as fits
        # within the token budget (oldest messages dropped first).
        payload = build_context(
            st.session_state.messages,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=MAX_TOKENS,
            max_turns=MAX_TURNS,
        )

        # Stream into a container, running a small tool-use loop: if the model
        # emits a "TOOL: ..." line, execute the tool, feed the result back, and
        # stream the final answer. Then apply the output guardrail (PII).
        with st.chat_message("assistant"):
            container = st.empty()
            work = list(payload)
            tool_notes = []
            final = ""
            start = time.perf_counter()
            for _ in range(MAX_TOOL_ROUNDS + 1):
                raw = ""
                for token in llm.stream(work):
                    raw += token
                    container.markdown(raw)
                call = parse_tool_call(raw)
                if not call:
                    final = raw
                    break
                name, arg = call
                result = run_tool(name, arg)
                tool_notes.append(f"`{name}({arg})` → {result}")
                work = work + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content":
                        f"TOOL_RESULT: {result}\nNow answer my original question using this result."},
                ]
            else:
                final = raw  # ran out of tool rounds; use the last output
            latency_ms = (time.perf_counter() - start) * 1000

            response, redactions = filter_output(final)
            container.markdown(response)
            if tool_notes:
                st.caption("🔧 Tools used: " + " ; ".join(tool_notes))
            if redactions:
                st.caption("⚠️ Redacted potential PII: " + ", ".join(redactions))

        st.session_state.messages.append({"role": "assistant", "content": response})
        log_turn(
            backend=llm.name,
            prompt_tokens=sum(estimate_tokens(m["content"]) for m in work),
            completion_tokens=estimate_tokens(response),
            latency_ms=latency_ms,
            redactions=redactions,
        )

# --- sidebar controls ---
with st.sidebar:
    st.header("Controls")
    st.write(f"Turns in memory: {len(st.session_state.messages) // 2}")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()
