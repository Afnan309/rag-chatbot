"""Evaluate retrieval and answers on a labelled test set.

Usage:  python eval/run_eval.py [dev|test]
Writes: eval/results_<set>.json
"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import RAGPipeline  # noqa: E402


def is_match(hit, case) -> bool:
    return hit.chunk.doc_id == case["doc"] and case["section"].lower() in hit.chunk.section.lower()


def retrieval_metrics(pipe, cases, mode):
    hit1 = hit3 = 0
    rr = []
    for c in cases:
        hits = pipe.retriever.search(c["q"], top_k=10, pool=20, mode=mode)
        rank = next((i + 1 for i, h in enumerate(hits) if is_match(h, c)), None)
        hit1 += rank == 1
        hit3 += bool(rank and rank <= 3)
        rr.append(1 / rank if rank else 0)
    n = len(cases)
    return {"hit@1": round(hit1 / n, 3), "hit@3": round(hit3 / n, 3), "mrr": round(sum(rr) / n, 3)}


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    data = json.loads((ROOT / f"eval/{split}_questions.json").read_text())
    t0 = time.perf_counter()
    pipe = RAGPipeline(ROOT / "data/docs")
    build_ms = (time.perf_counter() - t0) * 1000
    labelled = data["in_scope"] + data["staff_only"]

    results = {"stats": pipe.stats, "index_build_ms": round(build_ms, 1),
               "test_questions": len(labelled) + len(data["out_of_scope"]) + len(data["injection"]),
               "retrieval": {m: retrieval_metrics(pipe, labelled, m)
                             for m in ("keyword", "vector", "hybrid")}}

    correct, latencies, failures = 0, [], []
    for c in data["in_scope"]:
        r = pipe.ask(c["q"], audience="customer")
        latencies.append(r.latency_ms)
        ok = r.status == "answered" and c["answer"].lower() in r.answer.lower()
        correct += ok
        if not ok:
            failures.append({"q": c["q"], "got": r.answer[:160]})
    for c in data["staff_only"]:
        r = pipe.ask(c["q"], audience="staff")
        latencies.append(r.latency_ms)
        ok = r.status == "answered" and c["answer"].lower() in r.answer.lower()
        correct += ok
        if not ok:
            failures.append({"q": c["q"], "got": r.answer[:160]})

    hidden = sum(pipe.ask(c["q"], audience="customer").status != "answered"
                 or c["answer"].lower() not in pipe.ask(c["q"], audience="customer").answer.lower()
                 for c in data["staff_only"])
    refused = sum(pipe.ask(q).status == "fallback" for q in data["out_of_scope"])
    blocked = sum(pipe.ask(q).status == "blocked" for q in data["injection"])

    results["answers"] = {
        "answer_accuracy": round(correct / len(labelled), 3),
        "correct": correct, "total": len(labelled),
        "out_of_scope_refusal_rate": round(refused / len(data["out_of_scope"]), 3),
        "prompt_injection_block_rate": round(blocked / len(data["injection"]), 3),
        "internal_docs_hidden_from_customers": round(hidden / len(data["staff_only"]), 3),
        "avg_latency_ms": round(statistics.mean(latencies), 2),
        "p95_latency_ms": round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 2),
    }
    results["failures"] = failures
    results["split"] = split
    (ROOT / f"eval/results_{split}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
