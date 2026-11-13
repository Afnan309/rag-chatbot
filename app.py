"""Flask web app + REST API for the RAG customer-support chatbot.

Run:  python app.py   ->  http://localhost:8000
"""
import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from rag import RAGPipeline

ROOT = Path(__file__).parent
app = Flask(__name__)
pipeline = RAGPipeline(ROOT / "data" / "docs")


def _eval_summary() -> dict:
    path = ROOT / "eval" / "results_test.json"
    return json.loads(path.read_text()) if path.exists() else {}


@app.get("/")
def index():
    return render_template("index.html", stats=pipeline.stats, evals=_eval_summary())


@app.post("/api/chat")
def chat():
    body = request.get_json(force=True)
    question = (body.get("message") or "").strip()
    audience = body.get("audience", "customer")
    if not question:
        return jsonify({"error": "message is required"}), 400
    result = pipeline.ask(question, audience=audience)
    return jsonify({
        "answer": result.answer,
        "status": result.status,
        "sources": result.sources,
        "latency_ms": round(result.latency_ms, 2),
    })


@app.get("/api/stats")
def stats():
    return jsonify(pipeline.stats)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
