"""Create the demo agents (one per LLM provider) and load their sample documents.

    python3 scripts/seed_demo_agents.py                                  # local stack
    python3 scripts/seed_demo_agents.py --api https://chat.example.com/api --user admin

Standard library only, so it also runs on a fresh server. Safe to run again:
existing agents (matched by name) and documents (matched by file name) are
skipped.

The three agents exercise the three providers:
  * Agent HR                      -> OpenAI
  * Agent Legal                   -> LiteLLM (bundled proxy, routed to OpenAI)
  * Agen SDM (Bahasa Indonesia)   -> OpenRouter, with LiteLLM as its fallback.
    Without a valid OPENROUTER_API_KEY every answer shows the automatic fallback.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

SAMPLE_DATA = Path(__file__).resolve().parent.parent / "sample_data"

AGENTS = [
    {
        "name": "Agent HR",
        "description": "Answers employee questions from the employee handbook.",
        "instructions": (
            "You are the HR assistant for Nusantara Digital employees. Answer questions about "
            "leave, benefits, working hours and company policies using only the employee "
            "handbook. Be friendly and concise, and state exact numbers and deadlines."
        ),
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "documents": ["hr/employee_handbook.pdf"],
    },
    {
        "name": "Agent Legal",
        "description": "Contract templates and NDAs",
        "instructions": (
            "You are the legal assistant. Explain clauses from our contract templates and NDAs "
            "precisely. Quote clause numbers and defined terms, highlight obligations and "
            "deadlines, and never give advice beyond what the documents state."
        ),
        "provider": "litellm",
        "model": "gpt-4.1-mini",
        "documents": ["legal/service_agreement_template.docx", "legal/scanned_nda.pdf"],
    },
    {
        "name": "Agen SDM (Bahasa Indonesia)",
        "description": "Menjawab pertanyaan karyawan dari peraturan perusahaan (bahasa Indonesia).",
        "instructions": (
            "Anda adalah asisten SDM PT Nusantara Digital. Jawab dalam bahasa Indonesia "
            "berdasarkan peraturan perusahaan, sebutkan angka dan batas waktu secara tepat."
        ),
        "provider": "openrouter",
        "model": "openai/gpt-4.1-mini",
        "fallback_provider": "litellm",
        "fallback_model": "gpt-4.1-mini",
        "documents": ["hr_id/peraturan_perusahaan.pdf"],
    },
]


class Api:
    def __init__(self, base: str, auth: str | None) -> None:
        self.base = base.rstrip("/")
        self.headers = {}
        if auth:
            self.headers["Authorization"] = "Basic " + base64.b64encode(auth.encode()).decode()

    def call(self, method: str, path: str, body: bytes | None = None, content_type: str | None = None):
        headers = dict(self.headers)
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(self.base + path, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise SystemExit(f"{method} {path} failed: HTTP {exc.code} {detail}") from None
        except urllib.error.URLError as exc:
            raise SystemExit(f"Cannot reach {self.base}: {exc.reason}") from None

    def json(self, method: str, path: str, payload: dict | None = None):
        body = json.dumps(payload).encode() if payload is not None else None
        return self.call(method, path, body, "application/json")

    def upload(self, path: str, files: list[Path]):
        boundary = uuid.uuid4().hex
        parts = []
        for f in files:
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
                f"filename=\"{f.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode()
                + f.read_bytes()
                + b"\r\n"
            )
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        return self.call("POST", path, body, f"multipart/form-data; boundary={boundary}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8080/api")
    parser.add_argument("--user", help="basic-auth user for the production site (password is prompted)")
    args = parser.parse_args()
    auth = f"{args.user}:{getpass.getpass(f'Password for {args.user}: ')}" if args.user else None
    api = Api(args.api, auth)

    existing = {a["name"]: a for a in api.json("GET", "/agents")}
    for spec in AGENTS:
        documents = spec.pop("documents")
        agent = existing.get(spec["name"])
        if agent:
            print(f"= {spec['name']}: already exists ({agent['provider']} / {agent['model']})")
        else:
            agent = api.json("POST", "/agents", spec)
            print(f"+ {spec['name']}: created ({agent['provider']} / {agent['model']})")

        have = {d["filename"] for d in api.json("GET", f"/agents/{agent['id']}/documents")}
        missing = [SAMPLE_DATA / d for d in documents if Path(d).name not in have]
        if missing:
            result = api.upload(f"/agents/{agent['id']}/documents", missing)
            for r in result["rejected"]:
                print(f"  ! {r['filename']}: {r['message']}")
            print(f"  uploaded {', '.join(f.name for f in missing)}")

        deadline = time.time() + 300
        while time.time() < deadline:
            docs = api.json("GET", f"/agents/{agent['id']}/documents")
            if all(d["status"] != "processing" for d in docs):
                break
            time.sleep(2)
        for d in docs:
            print(f"  - {d['filename']}: {d['status']}" + (f" ({d['error_message']})" if d["error_message"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
