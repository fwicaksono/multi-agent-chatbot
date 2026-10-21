"""Evaluate retrieval and answer quality against a running deployment.

    cd backend
    uv run python eval/run_eval.py                       # against http://localhost:8080/api
    uv run python eval/run_eval.py --api http://localhost:8000/api --provider litellm

For each dataset in eval/datasets/ the script creates a fresh agent, uploads its
documents (from sample_data/), waits until they are ready, and then measures:

Retrieval (answerable questions, per search mode: vector / keyword / hybrid)
  * Hit@1, Hit@3, Hit@k - the expected document *and* page/section is in the top-k
  * MRR                 - mean reciprocal rank of the first correct passage

Answers (hybrid retrieval + the agent's LLM)
  * Answer accuracy     - every expected key fact appears in the answer
  * Citation accuracy   - the answer cites the expected document/location
  * Abstention accuracy - unanswerable / other-agent questions get "I don't know"
  * False abstentions   - answerable questions wrongly answered with "I don't know"
  * Multi-turn accuracy - follow-ups ("What about the NDA then?") resolved correctly

Results are written to eval/results/latest.json and latest.md. Agents are
deleted afterwards unless --keep is passed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
SAMPLE_DATA = HERE.parent.parent / "sample_data"
MODES = ("vector", "keyword", "hybrid")
IDK = re.compile(
    r"\b(don['’]t|do not|does not|doesn['’]t|cannot|can['’]t) (know|find|contain|mention|cover|"
    r"include|provide|specify|answer)|\bnot (covered|mentioned|specified|included|available)\b|"
    r"\bno (information|mention)\b",
    re.IGNORECASE,
)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("’", "'"))


def matches(source: dict[str, Any], expected: dict[str, Any]) -> bool:
    if source["document_name"] != expected["document"]:
        return False
    if "page" in expected and source.get("page_number") != expected["page"]:
        return False
    if "section" in expected and expected["section"] not in (source.get("section") or ""):
        return False
    return True


@dataclass
class Scores:
    retrieval: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {m: {"hit1": [], "hit3": [], "hitk": [], "rr": []} for m in MODES}
    )
    answer_correct: list[bool] = field(default_factory=list)
    citation_correct: list[bool] = field(default_factory=list)
    false_abstain: list[bool] = field(default_factory=list)
    abstain_correct: list[bool] = field(default_factory=list)
    multiturn_correct: list[bool] = field(default_factory=list)
    latency_ms: list[int] = field(default_factory=list)
    cost_usd: float = 0.0
    failures: list[dict[str, Any]] = field(default_factory=list)


def mean(xs: list[float] | list[bool]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


class Client:
    def __init__(self, base: str) -> None:
        self.http = httpx.Client(base_url=base.rstrip("/"), timeout=180)

    def call(self, method: str, path: str, **kw) -> Any:
        r = self.http.request(method, path, **kw)
        if r.status_code >= 400 and r.status_code != 502:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else None

    def ask(self, conversation_id: str, question: str) -> dict[str, Any]:
        return self.call(
            "POST",
            f"/conversations/{conversation_id}/messages",
            params={"stream": "false"},
            json={"content": question},
        )


def setup_agent(api: Client, dataset: dict, provider: str, model: str) -> str:
    agent = api.call(
        "POST", "/agents", json={**dataset["agent"], "provider": provider, "model": model}
    )
    files = [
        ("files", (Path(p).name, (SAMPLE_DATA / p).read_bytes(), "application/octet-stream"))
        for p in dataset["documents"]
    ]
    result = api.call("POST", f"/agents/{agent['id']}/documents", files=files)
    if result["rejected"]:
        raise RuntimeError(f"documents rejected: {result['rejected']}")
    deadline = time.time() + 300
    while time.time() < deadline:
        docs = api.call("GET", f"/agents/{agent['id']}/documents")
        if all(d["status"] != "processing" for d in docs):
            failed = [d for d in docs if d["status"] == "failed"]
            if failed:
                raise RuntimeError(f"ingestion failed: {[d['error_message'] for d in failed]}")
            return agent["id"]
        time.sleep(2)
    raise TimeoutError("documents were not ready within 5 minutes")


def evaluate(api: Client, dataset: dict, agent_id: str, top_k: int) -> Scores:
    s = Scores()
    for q in dataset["questions"]:
        answerable = q.get("answerable", True)
        if answerable:
            for mode in MODES:
                res = api.call(
                    "POST",
                    f"/agents/{agent_id}/retrieve",
                    json={"query": q["question"], "mode": mode, "top_k": top_k},
                )
                ranks = [
                    i for i, src in enumerate(res["sources"], 1) if matches(src, q["expected"])
                ]
                first = ranks[0] if ranks else None
                r = s.retrieval[mode]
                r["hit1"].append(float(first == 1))
                r["hit3"].append(float(first is not None and first <= 3))
                r["hitk"].append(float(first is not None))
                r["rr"].append(1.0 / first if first else 0.0)

        conv = api.call("POST", f"/agents/{agent_id}/conversations")
        reply = api.ask(conv["id"], q["question"])
        if reply.get("event") != "done":
            s.failures.append({"id": q["id"], "error": reply.get("message")})
            (s.answer_correct if answerable else s.abstain_correct).append(False)
            continue
        answer = reply["content"]
        s.latency_ms.append(reply.get("latency_ms") or 0)
        s.cost_usd += reply["usage"]["cost_usd"]
        abstained = bool(IDK.search(answer))

        if answerable:
            correct = all(norm(k) in norm(answer) for k in q["answer_keywords"])
            cited = any(matches(c, q["expected"]) for c in reply["citations"])
            s.answer_correct.append(correct)
            s.citation_correct.append(cited)
            s.false_abstain.append(abstained and not correct)
            if not (correct and cited):
                s.failures.append(
                    {"id": q["id"], "answer": answer[:200], "correct": correct, "cited": cited}
                )
        else:
            s.abstain_correct.append(abstained and not reply["citations"])
            if not abstained:
                s.failures.append({"id": q["id"], "answer": answer[:200], "expected": "abstain"})

    for c in dataset.get("conversations", []):
        conv = api.call("POST", f"/agents/{agent_id}/conversations")
        reply: dict[str, Any] = {}
        for turn in c["turns"]:
            reply = api.ask(conv["id"], turn)
        ok = reply.get("event") == "done" and all(
            norm(k) in norm(reply["content"]) for k in c["final_keywords"]
        )
        s.multiturn_correct.append(ok)
        if not ok:
            s.failures.append(
                {
                    "id": c["id"],
                    "answer": reply.get("content", "")[:200],
                    "rewritten": reply.get("rewritten_query"),
                }
            )
    return s


def summarise(name: str, s: Scores, n_questions: int) -> dict[str, Any]:
    return {
        "dataset": name,
        "questions": n_questions,
        "retrieval": {
            mode: {
                "hit@1": round(mean(v["hit1"]), 3),
                "hit@3": round(mean(v["hit3"]), 3),
                "hit@k": round(mean(v["hitk"]), 3),
                "mrr": round(mean(v["rr"]), 3),
            }
            for mode, v in s.retrieval.items()
        },
        "answer_accuracy": round(mean(s.answer_correct), 3),
        "citation_accuracy": round(mean(s.citation_correct), 3),
        "false_abstention_rate": round(mean(s.false_abstain), 3),
        "abstention_accuracy": round(mean(s.abstain_correct), 3),
        "multiturn_accuracy": round(mean(s.multiturn_correct), 3),
        "avg_latency_ms": int(mean(s.latency_ms)),
        "total_cost_usd": round(s.cost_usd, 5),
        "failures": s.failures,
    }


def to_markdown(results: list[dict], meta: dict) -> str:
    lines = [
        f"_Run {meta['date']} · provider `{meta['provider']}` · model `{meta['model']}` · "
        f"top-k {meta['top_k']}_",
        "",
        "**Retrieval** (answerable questions; a hit = expected document *and* page/section)",
        "",
        "| Agent | Mode | Hit@1 | Hit@3 | Hit@k | MRR |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        for mode in MODES:
            m = r["retrieval"][mode]
            lines.append(
                f"| {r['dataset']} | {mode} | {m['hit@1']:.2f} | {m['hit@3']:.2f} | "
                f"{m['hit@k']:.2f} | {m['mrr']:.2f} |"
            )
    lines += [
        "",
        "**Answers** (hybrid retrieval, end-to-end through the chat API)",
        "",
        "| Agent | Questions | Answer acc. | Citation acc. | Abstention acc. | False IDK | "
        "Multi-turn | Avg latency | Cost |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {r['questions']} | {r['answer_accuracy']:.0%} | "
            f"{r['citation_accuracy']:.0%} | {r['abstention_accuracy']:.0%} | "
            f"{r['false_abstention_rate']:.0%} | {r['multiturn_accuracy']:.0%} | "
            f"{r['avg_latency_ms'] / 1000:.1f}s | ${r['total_cost_usd']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--api", default="http://localhost:8080/api")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--dataset", action="append", help="dataset name(s); default: all")
    parser.add_argument("--keep", action="store_true", help="keep the evaluation agents")
    args = parser.parse_args()

    api = Client(args.api)
    paths = sorted((HERE / "datasets").glob("*.json"))
    if args.dataset:
        paths = [p for p in paths if p.stem in args.dataset]

    results = []
    for path in paths:
        dataset = json.loads(path.read_text())
        print(
            f"[{path.stem}] creating agent and ingesting {len(dataset['documents'])} document(s)..."
        )
        agent_id = setup_agent(api, dataset, args.provider, args.model)
        try:
            print(f"[{path.stem}] evaluating {len(dataset['questions'])} questions...")
            scores = evaluate(api, dataset, agent_id, args.top_k)
            results.append(summarise(path.stem, scores, len(dataset["questions"])))
        finally:
            if not args.keep:
                api.call("DELETE", f"/agents/{agent_id}")

    meta = {
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "provider": args.provider,
        "model": args.model,
        "top_k": args.top_k,
    }
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "latest.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    markdown = to_markdown(results, meta)
    (out / "latest.md").write_text(markdown)
    print("\n" + markdown)
    for r in results:
        for f in r["failures"]:
            print(f"  ! {r['dataset']} {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
