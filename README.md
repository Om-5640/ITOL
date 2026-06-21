# ITOL — Intelligent Token Optimization Layer

ITOL is a self-hosted Python middleware that sits between your application and any LLM API. It reduces token costs by 20–60 % on real workloads through seven composable optimization strategies, while providing a machine-verifiable quality guarantee on every dispatched prompt via the Quality Preservation Score (QPS) gate.

---

## Architecture

```
Your App / LangChain / SDK
         │
         ▼  POST /v1/chat/completions  (OpenAI dialect)
    ┌─────────────────────────────────────────────────────┐
    │                  ITOL Proxy                          │
    │                                                      │
    │  Ingest & Analyse  ──►  S2 Instruction Compress      │
    │  (segment, classify,     S1 Semantic Dedupe          │
    │   manifest extract)      S6 Structural Minify        │
    │                          S3 Context Window           │
    │                          S5 Convo Distil             │
    │  L0 Exact Cache  ──────► S4 RACR Doc Replace         │
    │  L1 Semantic Cache       S7 Lossy Token Compress     │
    │  L2 Template Cache  │                                │
    │                     ▼                                │
    │            QPS Gate (manifest coverage × semantic    │
    │            fidelity × min-window fidelity)           │
    │              │ pass              │ fail              │
    │              ▼                   ▼                   │
    │        Dispatch optimized    Rollback → dispatch raw │
    └───────────────────┬─────────────────────────────────┘
                        │
                        ▼
               Upstream LLM API
          (OpenAI / Anthropic / Mistral
           Groq / Ollama / Cohere / any)
```

---

## Quickstart

```bash
pip install itol

# Calibrate on synthetic data (no internet required)
itol calibrate --offline

# Start the proxy
itol serve --port 8787 --upstream https://api.openai.com/v1/chat/completions

# Open the dashboard
open http://localhost:8787/dashboard
```

---

## Integration Examples

### Python SDK (drop-in replacement)

```python
import openai

client = openai.OpenAI(
    api_key="sk-...",
    base_url="http://localhost:8787/v1",   # ← point at ITOL
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarise this 10 000-token document..."}],
)

# Response headers carry ITOL metadata
# X-ITOL-Saved-Tokens: 3847
# X-ITOL-Cache: l1
# X-ITOL-QPS: 0.994
```

### Direct proxy (curl)

```bash
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_base="http://localhost:8787/v1",
    openai_api_key="sk-...",
)
```

---

## Adding a New Provider (< 30 minutes)

```bash
itol new-adapter --name MyProvider
# → itol/adapters/myprovider.py (ready to fill in base_url + capabilities)
```

For OpenAI-compatible APIs (Mistral, Groq, Ollama, …) the entire adapter is ~10 lines:

```python
from itol.adapters.openai_compatible_base import OpenAICompatibleAdapter

class MyProviderAdapter(OpenAICompatibleAdapter):
    _name    = "myprovider"
    base_url = "https://api.myprovider.com/v1"

    def capabilities(self):
        return {"native_prompt_cache": "none", "cache_read_discount": 0.0, "max_context": 32_768}
```

Register it in `pyproject.toml`:

```toml
[project.entry-points."itol.adapters"]
myprovider = "itol.adapters.myprovider:MyProviderAdapter"
```

---

## Optimization Strategies

| ID | Name | Type | What it does |
|----|------|------|-------------|
| S1 | Semantic Dedupe | Lossless | Removes near-duplicate segments (MinHash + Jaccard) |
| S2 | Instruction Compress | Lossless | Template mining; re-uses compressed system instructions |
| S3 | Context Window | Near-lossless | Relevance-ranked truncation (mass floor ≥ 0.90–0.97) |
| S4 | RACR | Near-lossless | Retrieval-augmented context replacement for long docs |
| S5 | Convo Distil | Lossy-bounded | Distils turns older than K=6 into a structured ledger |
| S6 | Structural Minify | Lossless | Whitespace, tool-schema, trajectory hygiene |
| S7 | Lossy Compress | Lossy | LLMLingua-2 ONNX; off by default — opt-in per class |

---

## CLI Reference

```
itol calibrate [--offline] [--online] [--n-synth N]
itol status
itol serve [--port PORT] [--host HOST] [--upstream URL] [--reload]
itol new-adapter --name NAME [--output DIR]
```

---

## Dashboard

Live at `http://localhost:8787/dashboard` — auto-updates via SSE:

- Cost saved (USD) and tokens saved with sparklines
- QPS distribution histogram with 0.98 floor annotation
- L0 / L1 / L2 cache hit-rate donuts
- Strategy breakdown horizontal bar chart
- Real-time request activity feed

---

## Testing

```bash
pip install -e ".[dev]"
pytest                    # 1161+ tests, ~8 s
pytest tests/e2e/         # 9 end-to-end integration scenarios
```

---

## Deployment

### Docker

```bash
docker build -t itol:latest -f deploy/Dockerfile .
docker run -p 8787:8000 -v itol-data:/data itol:latest
```

### Kubernetes / Helm

```bash
helm install itol deploy/helm/ \
  --set image.repository=your-registry/itol \
  --set image.tag=latest
```

---

## Hard Constraints

1. **Zero external services by default** — SQLite cache, in-process vector index, no Redis / Qdrant required.
2. **Never returns 500** (CR-16) — pipeline exceptions fall back to raw dispatch.
3. **Quality gate on every dispatch** — manifest coverage = 1.0 enforced before optimised prompt leaves the proxy.
4. **Self-hosted** — your prompts never leave your infrastructure.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
