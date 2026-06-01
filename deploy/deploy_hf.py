"""Deploy the OSS assistant to a Hugging Face Streamlit Space.

Usage:
    pip install huggingface_hub
    python deploy/deploy_hf.py --space <user>/os-assistant --token hf_xxx

Creates (or reuses) a public Streamlit Space and uploads the app, its support
packages, and a Space requirements.txt (the contents of requirements-hf.txt).
The app auto-detects the Space (SPACE_ID) and runs BACKEND=hf, serving
Qwen2.5-0.5B-Instruct on CPU via transformers. Get a write token at
https://huggingface.co/settings/tokens.
"""
import argparse
import os
import shutil
import tempfile

SPACE_README = """---
title: OS Assistant (OSS)
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# OS Assistant — Open Source Model

Qwen2.5-0.5B-Instruct running on CPU via transformers, with short-term memory
and safety guardrails. Part of an OSS-vs-frontier assistant comparison; see the
GitHub repository for the full project, evaluation harness, and report.
"""

INCLUDE_FILES = ["app.py"]
INCLUDE_DIRS = ["llm", "memory", "safety", "observability", "tools"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="e.g. username/os-assistant")
    ap.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF write token")
    args = ap.parse_args()
    if not args.token:
        raise SystemExit("Provide --token or set HF_TOKEN")

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.space, repo_type="space",
                    space_sdk="streamlit", exist_ok=True)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as tmp:
        for f in INCLUDE_FILES:
            shutil.copy(os.path.join(root, f), os.path.join(tmp, f))
        for d in INCLUDE_DIRS:
            shutil.copytree(os.path.join(root, d), os.path.join(tmp, d),
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # The Space installs from a file literally named requirements.txt,
        # so the HF deploy deps become that file.
        shutil.copy(os.path.join(root, "requirements-hf.txt"),
                    os.path.join(tmp, "requirements.txt"))
        with open(os.path.join(tmp, "README.md"), "w", encoding="utf-8") as fh:
            fh.write(SPACE_README)
        api.upload_folder(folder_path=tmp, repo_id=args.space, repo_type="space")

    print(f"Deployed: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
