"""Ollama adapter for the SENTINEL-EDGE analyst.

Wraps a local Ollama model as the `llm` callable that
analyst.render_shift_report expects: prompt string in, text out.

Requires Ollama serving (default http://localhost:11434) with a pulled
model, e.g. `ollama pull qwen3:8b`. Nothing leaves the machine.
"""

import json
import urllib.request


def ollama_llm(prompt: str, model: str = "qwen3:8b",
               host: str = "http://localhost:11434",
               timeout: int = 120) -> str:
    req = urllib.request.Request(
        host + "/api/generate",
        data=json.dumps({"model": model, "prompt": prompt,
                         "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"].strip()


if __name__ == "__main__":
    import sys
    from analyst.analyst import render_shift_report
    from eventdb.eventdb import EventStore
    db = sys.argv[1] if len(sys.argv) > 1 else "sentinel.db"
    store = EventStore(db)
    print(render_shift_report(store.shift_summary(),
                              llm=ollama_llm))
    store.close()
