"""
FAQ cache workload.

50 base queries + 100 paraphrases (2 per base) = 150 total.
The headline metric for this workload is L0 + L1 cache hit rate.

After the first run, identical or near-identical queries should hit L0 (exact)
or L1 (semantic), producing 60–80% cache hit rate.

The ITOL runner processes queries in order: base queries first, then paraphrases.
The baseline runner always dispatches (no cache).
"""
from __future__ import annotations

import hashlib
import random

from bench.workloads import WorkloadSample

# ---------------------------------------------------------------------------
# FAQ base topics
# ---------------------------------------------------------------------------

_BASE_FAQS = [
    # (question, short_answer_hint)
    ("What is the difference between RAM and ROM?", "RAM is volatile; ROM is non-volatile"),
    ("How does HTTPS encryption work?", "TLS handshake with asymmetric then symmetric keys"),
    ("What is a REST API?", "HTTP-based stateless interface with CRUD operations"),
    ("Explain the CAP theorem.", "Consistency, Availability, Partition-tolerance — pick two"),
    ("What is a docker container?", "Isolated process namespace with layered filesystem"),
    ("How does garbage collection work in Python?", "Reference counting + cyclic GC"),
    ("What is the difference between SQL and NoSQL?", "Relational schema vs flexible documents"),
    ("Explain Big O notation.", "Upper bound on algorithm time complexity"),
    ("What is a neural network?", "Layered weighted nodes trained by gradient descent"),
    ("How does DNS resolution work?", "Recursive query to root → TLD → authoritative nameserver"),
    ("What is machine learning?", "Algorithms that improve with data without explicit programming"),
    ("Explain the OSI model.", "7-layer networking stack from physical to application"),
    ("What is a hash function?", "Deterministic one-way mapping to fixed-size digest"),
    ("How does public key cryptography work?", "Key pair: encrypt with public, decrypt with private"),
    ("What is microservices architecture?", "Small independent services communicating via APIs"),
    ("Explain eventual consistency.", "Distributed system converges to same state over time"),
    ("What is a binary search tree?", "BST: left < root < right, O(log n) search"),
    ("How does TCP/IP work?", "Connection-oriented, ordered, error-checked data delivery"),
    ("What is continuous integration?", "Automated build+test on every code commit"),
    ("Explain the difference between process and thread.", "Thread shares memory; process has own space"),
    ("What is a Kubernetes pod?", "Smallest deployable unit containing one or more containers"),
    ("How does load balancing work?", "Distributes traffic across multiple servers by algorithm"),
    ("What is OAuth 2.0?", "Authorization framework delegating access via tokens"),
    ("Explain database indexing.", "B-tree structure enabling O(log n) lookups"),
    ("What is a message queue?", "Async buffer decoupling producers from consumers"),
    ("How does SSL certificate validation work?", "Chain of trust from root CA through intermediaries"),
    ("What is A/B testing?", "Randomly split traffic to compare variant performance"),
    ("Explain MapReduce.", "Parallel processing: map to key-value pairs, reduce to aggregate"),
    ("What is a CDN?", "Geographically distributed servers caching static assets close to users"),
    ("How does a relational database join work?", "Merge matching rows across tables by key"),
    ("What is the actor model in concurrency?", "Actors communicate by message passing, no shared state"),
    ("Explain serverless computing.", "Event-driven execution without managing server infrastructure"),
    ("What is a virtual machine?", "Emulated computer running on physical hardware via hypervisor"),
    ("How does Bloom filter work?", "Probabilistic set membership with tunable false positive rate"),
    ("What is the two-phase commit protocol?", "Coordinator asks all nodes to prepare, then commit or abort"),
    ("Explain backpressure in streaming systems.", "Downstream signals upstream to slow production"),
    ("What is observability in software?", "Metrics, logs, traces enabling system state inference"),
    ("How does a LRU cache work?", "Evicts least-recently-used entry when capacity exceeded"),
    ("What is dependency injection?", "Providing dependencies externally rather than constructing them"),
    ("Explain the SOLID principles.", "Single responsibility, Open-closed, Liskov, Interface, Dependency"),
    ("What is a merkle tree?", "Hash tree where each node is hash of children; used in blockchain"),
    ("How does a consensus algorithm work?", "Distributed agreement protocol like Raft or Paxos"),
    ("What is a GraphQL query?", "Typed schema query language letting clients request exact fields"),
    ("Explain eventual vs strong consistency.", "Strong: reads always see latest write; eventual: converges later"),
    ("What is sharding in databases?", "Horizontal partitioning distributing rows across multiple nodes"),
    ("How does a recommendation engine work?", "Collaborative or content-based filtering on user-item matrix"),
    ("What is a circuit breaker pattern?", "Fails fast when downstream error rate exceeds threshold"),
    ("Explain feature flags.", "Runtime toggles enabling gradual rollout without deployments"),
    ("What is a webhook?", "HTTP callback pushed from server to client on event"),
    ("How does rate limiting work?", "Token bucket or sliding window throttling requests per client"),
]

# ---------------------------------------------------------------------------
# Paraphrase templates
# ---------------------------------------------------------------------------

_PARAPHRASE_PATTERNS = [
    "Can you explain {base_no_q}",
    "I'd like to understand {base_no_q}",
    "Could you clarify {base_no_q}",
    "Help me understand: {base_no_q}",
    "What does {key_term} mean and how does it work?",
    "I've heard of {key_term} — can you give me a clear explanation?",
    "In simple terms, {base_lower}",
    "For a beginner: {base_lower}",
]


def _extract_key_term(question: str) -> str:
    """Extract the main technical term from a question."""
    # Remove question words and punctuation, take first meaningful phrase
    q = question.rstrip("?").strip()
    for prefix in ["What is ", "What are ", "How does ", "How do ", "Explain "]:
        if q.startswith(prefix):
            return q[len(prefix):]
    return q.split(",")[0].strip()


def _make_paraphrase(base_question: str, idx: int, rng: random.Random) -> str:
    base_no_q = base_question.rstrip("?").lstrip("What is ").lstrip("How does ").lstrip("Explain ").strip()
    key_term = _extract_key_term(base_question)
    pattern = _PARAPHRASE_PATTERNS[idx % len(_PARAPHRASE_PATTERNS)]
    para = pattern.format(
        base_no_q=base_no_q.lower(),
        base_lower=base_question.lower(),
        key_term=key_term,
    )
    # Capitalize first letter
    return para[0].upper() + para[1:] if para else base_question


# ---------------------------------------------------------------------------
# Build WorkloadSamples
# ---------------------------------------------------------------------------

def _faq_to_sample(
    question: str,
    hint: str,
    base_id: str,
    paraphrase_of: str | None,
    seed: int,
) -> WorkloadSample:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful technical assistant. Provide clear, concise explanations.",
        },
        {"role": "user", "content": question},
    ]
    sid = hashlib.sha256(f"faq_{question}".encode()).hexdigest()[:16]
    return WorkloadSample(
        sample_id=f"faq_{sid}",
        workload="faq",
        messages=messages,
        gold_answer=hint,
        paraphrase_of=paraphrase_of,
        metadata={
            "base_id": base_id,
            "is_paraphrase": paraphrase_of is not None,
        },
    )


def load_faq_samples(n: int = 150, seed: int = 42) -> list[WorkloadSample]:
    """
    Generate n FAQ samples: base queries first, then paraphrases.
    Default n=150 → 50 base + 100 paraphrases.
    """
    rng = random.Random(seed)
    n_base = min(50, n // 3)
    n_para = n - n_base

    base_faqs = _BASE_FAQS[:n_base]
    samples: list[WorkloadSample] = []

    # Add base queries first (cache misses on first run)
    for i, (question, hint) in enumerate(base_faqs):
        bid = f"base_{i:03d}"
        samples.append(_faq_to_sample(question, hint, bid, None, seed))

    # Add paraphrases (should hit L0/L1 on ITOL run)
    para_per_base = max(1, n_para // n_base)
    for i, (question, hint) in enumerate(base_faqs):
        bid = f"base_{i:03d}"
        for j in range(para_per_base):
            if len(samples) >= n:
                break
            para_q = _make_paraphrase(question, j, rng)
            samples.append(_faq_to_sample(para_q, hint, bid, bid, seed))
        if len(samples) >= n:
            break

    return samples[:n]
