# OS Assistant — OSS vs Frontier Personal Assistant

A simple personal assistant built **once** and run on two different model
tiers — a small **open-source** model (Qwen2.5 via Ollama) and a large
**hosted frontier-tier** model (Llama-3.3-70B via Groq) — then evaluated
side-by-side on **hallucination, bias, and content safety**.

The whole thing is one Streamlit app with a pluggable model backend, a
short-term memory layer, a deterministic safety/guardrail layer, per-turn
observability logging, and an LLM-as-judge evaluation harness.

---

## ✨ Features

- **Multi-turn chat** with streaming responses (Streamlit)
- **Short-term memory** — token-aware sliding window of recent turns
- **Safety guardrails** — input blocklist (refuses harmful requests) + output PII redaction + safety system prompt
- **Tool use** — backend-agnostic `TOOL:` protocol with a safe calculator and a current-date tool (ReAct-style loop)
- **Pluggable backends** — `ollama` (local OSS) · `hf` (deploy) · `groq` (frontier) · `gemini` (optional) behind one interface
- **Observability** — every turn logged to JSONL (backend, tokens, latency, guardrail hits)
- **Evaluation harness** — runs both assistants over factual / jailbreak / bias prompts and scores them with an LLM-as-judge
- **Report generator** — infographics + cost/latency table + 1-page Markdown report

## 🏗️ Architecture

```
                 Streamlit UI (app.py)
                        │  messages[]
                        ▼
         Guardrails (safety/guardrails.py)
        input blocklist · system prompt · PII redaction
                        │
                        ▼
            Memory (memory/context.py)
       system prompt + recent turns under a token budget
                        │
                        ▼
          Backend factory  llm/get_llm()
   ┌──────────┬──────────────┬──────────────┐
 ollama       hf            groq           gemini
 Qwen2.5    Qwen2.5-0.5B   Llama-70B      (optional)
 (local)    (HF Space)     (frontier)
                        │ every turn
                        ▼
        Observability (observability/logger.py)  →  turns.jsonl
```

**One app, two assistants.** The OSS and Frontier "assistants" are the *same*
codebase — only the backend differs (`BACKEND` env var). This guarantees the
comparison is apples-to-apples and that both have identical capabilities
(memory, guardrails, UI).

## 📦 Project structure

```
app.py                  Streamlit chat UI
llm/                    backends + get_llm() factory (ollama / hf / groq / gemini)
memory/context.py       token-aware short-term memory
safety/guardrails.py    input blocklist + PII redaction + safety prompt
tools/toolbox.py        tool-use protocol + calculator/now tools
observability/logger.py per-turn JSONL logging
eval/                   prompt sets, run_eval.py, judge.py, results/
reports/                make_report.py + charts + evaluation_report.md
requirements.txt        local deps (no torch)
requirements-hf.txt     HF Spaces deploy deps (transformers + torch)
```

---

## 🚀 Setup (local)

**Prerequisites:** Python 3.12, [Ollama](https://ollama.com/download).

```bash
# 1. Install and pull the OSS model
ollama pull qwen2.5:1.5b

# 2. Create a virtual environment and install deps
python -m venv .venv
.venv\Scripts\activate            # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# 3. Configure
copy .env.example .env            # Windows  (use: cp on macOS/Linux)

# 4. Run the assistant
streamlit run app.py
```

Open http://localhost:8501.

### Run on the frontier model
Get a free key at [console.groq.com](https://console.groq.com), add it to `.env`:
```
GROQ_API_KEY=gsk_...
```
Then set `BACKEND=groq` in `.env` and re-run. Same app, frontier model.

## 🧪 Evaluation

```bash
python -m eval.run_eval       # runs both assistants + judges them -> eval/results/
python -m reports.make_report # builds charts + 1-page report -> reports/
```

See **[reports/evaluation_report.md](reports/evaluation_report.md)** for the full report.

### Results summary

| Metric | OSS (Qwen2.5-1.5B) | Frontier (Llama-3.3-70B) |
|---|---|---|
| Hallucination rate *(lower better)* | 10% | 0% |
| Content safety *(higher better)* | 62% | 100% |
| Bias-free rate *(higher better)* | 75% | 100% |

| Backend | Cost | Avg warm latency |
|---|---|---|
| OSS (local CPU, Ollama) | $0 | ~1035 ms |
| Frontier (Groq API) | $0 free tier (~$0.00012/turn paid) | ~468 ms |

---

## ☁️ Deploy the OSS model (Hugging Face Spaces)

The deployed copy uses the `hf` backend (Qwen2.5-0.5B-Instruct via `transformers`
on CPU) because free Spaces can't run Ollama. The app auto-detects the Space
(`SPACE_ID` env var) and defaults to `BACKEND=hf` — no manual config needed.

**Easiest (scripted):**
```bash
pip install huggingface_hub
python deploy/deploy_hf.py --space <your-username>/os-assistant --token hf_xxx
```
This creates a public Streamlit Space and uploads the app + the deploy
requirements. (Get a write token at huggingface.co/settings/tokens.)

**Manual:** create a Streamlit Space at huggingface.co/new-space, then upload
`app.py`, `llm/`, `memory/`, `safety/`, `observability/`, and a
`requirements.txt` containing the contents of `requirements-hf.txt`.

---

## 🧭 Architecture decisions & tradeoffs

- **Pluggable backend, one app.** Satisfies both the OSS and frontier
  requirements with identical capabilities; swapping models is one env var.
- **Ollama locally, transformers on deploy.** Best dev experience (fast,
  offline, no keys) while still deploying free on HF Spaces; lazy imports keep
  the local install lightweight (no torch).
- **Groq for the frontier tier.** Anthropic/OpenAI/Gemini free tiers were
  unusable (paid or ~20 req/day). Groq's free tier (~30/min, thousands/day)
  makes the eval **free and re-runnable**. Llama-3.3-70B is a large open model
  served via a hosted API — equivalent to the brief's "DeepSeek" example.
- **Deterministic guardrails + model safety.** Simple, transparent regex rules
  as a backstop that complements the model's own training.
- **Token-window memory, no vector DB.** "Short-term memory" only needs the
  recent conversation resent each turn; a vector DB would be over-engineering.

## 🔭 What I'd improve with more time

- Larger, categorized eval set (100+ prompts) + a **second independent judge**
  to remove self-judging bias and report confidence intervals.
- Swap the regex guardrails for an **ML moderation classifier**.
- **Persistent memory** (SQLite) and **more tools** (web search, code execution) — a calculator + date tool are already implemented.
- A live **observability dashboard** (read `turns.jsonl`) and CI that runs the
  eval on each change.
- Quantized/GPU serving for the deployed model to cut latency.

## License

MIT (for the assessment).
