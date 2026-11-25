from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

REPO_ROOT = Path(__file__).resolve().parent.parent
RAG_AGENTS_DIR = REPO_ROOT / "rag_agents"
if str(RAG_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_AGENTS_DIR))

from rag.config import COLLECTION_NAME, EMBEDDING_MODEL, PERSIST_DIR  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_GOLDEN_PATH = EVAL_DIR / "golden_copom_eval.json"


@dataclass
class EvalQuestion:
    id: str
    category: str
    question: str
    meetings: list[str]
    expected_meetings: list[str]
    evidence_patterns: list[str]
    gold_chunk_ids: list[str] = field(default_factory=list)


DEFAULT_EVAL_QUESTIONS: list[EvalQuestion] = [
    # --- Group A: Exact Terminology & Meeting Anchors (5 queries) ---
    EvalQuestion(
        id="Q01",
        category="exact_terminology",
        question="Qual foi a decisão de política monetária sobre a taxa Selic na 282ª reunião?",
        meetings=["282"],
        expected_meetings=["282"],
        evidence_patterns=["d) decisao de politica monetaria"],
    ),
    EvalQuestion(
        id="Q02",
        category="exact_terminology",
        question="Qual foi a decisão de política monetária sobre a taxa Selic na 280ª reunião?",
        meetings=["280"],
        expected_meetings=["280"],
        evidence_patterns=["d) decisao de politica monetaria"],
    ),
    EvalQuestion(
        id="Q03",
        category="exact_terminology",
        question="Quais eram as expectativas de inflação apuradas pela pesquisa Focus para 2026 e 2027?",
        meetings=["280", "281", "282"],
        expected_meetings=["280", "281", "282"],
        evidence_patterns=["pesquisa focus", "2026"],
    ),
    EvalQuestion(
        id="Q04",
        category="exact_terminology",
        question="O que o Comitê destacou sobre a convergência da inflação à meta no horizonte relevante?",
        meetings=["279", "280", "281", "282"],
        expected_meetings=["279", "280", "281", "282"],
        evidence_patterns=["horizonte relevante", "meta"],
    ),
    EvalQuestion(
        id="Q05",
        category="exact_terminology",
        question="A decisão de política monetária na 281ª reunião foi aprovada por unanimidade?",
        meetings=["281"],
        expected_meetings=["281"],
        evidence_patterns=["votaram", "decisao"],
    ),
    # --- Group B: Paraphrased / Vocabulary-Mismatch Queries (6 queries) ---
    EvalQuestion(
        id="Q06",
        category="paraphrased_semantic",
        question="O que o mercado financeiro projetava para o IPCA de 2026 e 2027 nas atas mais recentes?",
        meetings=["280", "281", "282"],
        expected_meetings=["280", "281", "282"],
        evidence_patterns=["pesquisa focus", "2026"],
    ),
    EvalQuestion(
        id="Q07",
        category="paraphrased_semantic",
        question="Quais fatores poderiam pressionar os preços para cima além do cenário base?",
        meetings=["280", "281", "282"],
        expected_meetings=["280", "281", "282"],
        evidence_patterns=["riscos de alta"],
    ),
    EvalQuestion(
        id="Q08",
        category="paraphrased_semantic",
        question="O que poderia fazer a inflação desacelerar mais rápido do que o previsto pelo Banco Central?",
        meetings=["280", "281", "282"],
        expected_meetings=["280", "281", "282"],
        evidence_patterns=["riscos de baixa"],
    ),
    EvalQuestion(
        id="Q09",
        category="paraphrased_semantic",
        question="Como os gastos públicos e o equilíbrio das contas do governo afetam o prêmio de risco e os juros?",
        meetings=["279", "280", "281", "282"],
        expected_meetings=["279", "280", "281", "282"],
        evidence_patterns=["fiscal", "premio de risco"],
    ),
    EvalQuestion(
        id="Q10",
        category="paraphrased_semantic",
        question="Como as tensões geopolíticas e os conflitos internacionais influenciaram o cenário global?",
        meetings=["278", "279", "280"],
        expected_meetings=["278", "279", "280"],
        evidence_patterns=["ambiente externo", "conflitos"],
    ),
    EvalQuestion(
        id="Q11",
        category="paraphrased_semantic",
        question="Como mudanças na jornada de trabalho e eventos climáticos impactam os custos na reunião 282?",
        meetings=["282"],
        expected_meetings=["282"],
        evidence_patterns=["escala 6x1"],
    ),
    # --- Group C: Hybrid Multi-Constraint & Cross-Meeting Queries (4 queries) ---
    EvalQuestion(
        id="Q12",
        category="hybrid_multi_meeting",
        question="Compare a taxa de juros definida nas reuniões 278, 280 e 282.",
        meetings=["278", "280", "282"],
        expected_meetings=["278", "280", "282"],
        evidence_patterns=["d) decisao de politica monetaria"],
    ),
    EvalQuestion(
        id="Q13",
        category="hybrid_multi_meeting",
        question="Por que o aquecimento do emprego e dos salários exige juros contracionistas por mais tempo?",
        meetings=["278", "279", "280"],
        expected_meetings=["278", "279", "280"],
        evidence_patterns=["mercado de trabalho", "servicos"],
    ),
    EvalQuestion(
        id="Q14",
        category="hybrid_multi_meeting",
        question="Como o aperto monetário e o crédito estão desacelerando a economia nas atas de 2025 e 2026?",
        meetings=["276", "278", "280", "282"],
        expected_meetings=["276", "278", "280", "282"],
        evidence_patterns=["transmissao da politica monetaria"],
    ),
    EvalQuestion(
        id="Q15",
        category="hybrid_multi_meeting",
        question="Quais foram as projeções de inflação do Comitê no cenário de referência?",
        meetings=["276", "277", "278", "279", "280", "281", "282"],
        expected_meetings=["276", "277", "278", "279", "280", "281", "282"],
        evidence_patterns=["debrucou entao sobre as projecoes"],
    ),
]


def normalize_for_match(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", without_accents).strip()


def chunk_key(doc: Document) -> str:
    return f"{doc.metadata['meeting']}:{doc.metadata['start_index']}"


def load_vector_store() -> Chroma:
    persist_path = RAG_AGENTS_DIR / PERSIST_DIR.lstrip("/")
    if not persist_path.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at {persist_path}. "
            "Run `python rag_agents/rag/ingest_copom.py` first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )


def load_all_indexed_docs(vector_store: Chroma) -> list[Document]:
    raw = vector_store.get(include=["documents", "metadatas"])
    docs: list[Document] = []
    for content, meta in zip(raw.get("documents") or [], raw.get("metadatas") or []):
        if meta and "meeting" in meta and "start_index" in meta:
            docs.append(Document(page_content=content, metadata=dict(meta)))
    if not docs:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' is empty. Run `ingest_copom.py` before benchmarking."
        )
    return docs


def resolve_gold_chunks(
    item: EvalQuestion,
    indexed_docs: list[Document],
    max_auto_gold: int = 3,
) -> list[str]:
    """Use pinned gold_chunk_ids if present; otherwise resolve top matches across expected meetings."""
    valid_keys = {chunk_key(d) for d in indexed_docs}
    pinned = [cid for cid in item.gold_chunk_ids if cid in valid_keys]
    if pinned:
        return pinned

    normalized_patterns = [normalize_for_match(p) for p in item.evidence_patterns]
    best_per_meeting: dict[str, tuple[int, str]] = {}

    for doc in indexed_docs:
        meeting = str(doc.metadata["meeting"])
        if meeting not in item.expected_meetings:
            continue
        norm_text = normalize_for_match(doc.page_content)
        if all(pat in norm_text for pat in normalized_patterns):
            term_hits = sum(norm_text.count(pat) for pat in normalized_patterns)
            candidate = (term_hits, chunk_key(doc))
            if (
                meeting not in best_per_meeting
                or candidate[0] > best_per_meeting[meeting][0]
            ):
                best_per_meeting[meeting] = candidate

    ranked = sorted(best_per_meeting.values(), key=lambda pair: (-pair[0], pair[1]))
    return [cid for _, cid in ranked[:max_auto_gold]]


def deduplicate_keys(docs: list[Document]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for doc in docs:
        key = chunk_key(doc)
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def retrieve_dense(
    vector_store: Chroma, question: str, meetings: list[str], k: int = 6
) -> list[Document]:
    where_filter = {"meeting": {"$in": meetings}}
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,
            "lambda_mult": 0.5,
            "filter": where_filter,
        },
    )
    return retriever.invoke(question)


def retrieve_bm25(
    scoped_docs: list[Document], question: str, k: int = 6
) -> list[Document]:
    if not scoped_docs:
        return []
    bm25 = BM25Retriever.from_documents(documents=scoped_docs, k=k)
    return bm25.invoke(question)


def retrieve_hybrid(
    vector_store: Chroma,
    scoped_docs: list[Document],
    question: str,
    meetings: list[str],
    top_k: int = 6,
) -> list[Document]:
    """Matches rag/chain_copom.py production retriever configuration (k=6)."""
    if not scoped_docs:
        return []
    where_filter = {"meeting": {"$in": meetings}}
    bm25_retriever = BM25Retriever.from_documents(documents=scoped_docs, k=6)
    dense_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.5,
            "filter": where_filter,
        },
    )
    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )
    return ensemble.invoke(question)[:top_k]


def compute_query_metrics(
    retrieved_keys: list[str], gold_keys: list[str]
) -> dict[str, float]:
    gold_set = set(gold_keys)
    if not gold_set:
        return {
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@6": 0.0,
            "hit@6": 0.0,
            "mrr@6": 0.0,
        }

    top1 = set(retrieved_keys[:1])
    top3 = set(retrieved_keys[:3])
    top6 = set(retrieved_keys[:6])

    mrr = 0.0
    for rank, key in enumerate(retrieved_keys[:6], start=1):
        if key in gold_set:
            mrr = 1.0 / rank
            break

    return {
        "recall@1": len(top1 & gold_set) / len(gold_set),
        "recall@3": len(top3 & gold_set) / len(gold_set),
        "recall@6": len(top6 & gold_set) / len(gold_set),
        "hit@6": 1.0 if (top6 & gold_set) else 0.0,
        "mrr@6": mrr,
    }


def load_or_init_questions(golden_path: Path | None) -> list[EvalQuestion]:
    if golden_path is not None and golden_path.exists():
        raw_items = json.loads(golden_path.read_text(encoding="utf-8"))
        return [
            EvalQuestion(**{k: v for k, v in item.items() if not k.startswith("_")})
            for item in raw_items
        ]
    return DEFAULT_EVAL_QUESTIONS


def export_bootstrap_golden(
    questions: list[EvalQuestion],
    indexed_docs: list[Document],
    output_path: Path,
) -> None:
    doc_lookup = {chunk_key(d): d for d in indexed_docs}
    payload = []
    for item in questions:
        resolved = resolve_gold_chunks(item, indexed_docs)
        entry = asdict(item)
        entry["gold_chunk_ids"] = resolved
        entry["_preview_snippets"] = [
            {
                "chunk_id": cid,
                "snippet": " ".join(doc_lookup[cid].page_content.split())[:180],
            }
            for cid in resolved
            if cid in doc_lookup
        ]
        payload.append(entry)

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Exported {len(payload)} golden queries to {output_path}")


def run_benchmark(
    golden_path: Path | None,
    results_path: Path | None,
    bootstrap: bool,
) -> None:
    vector_store = load_vector_store()
    indexed_docs = load_all_indexed_docs(vector_store)
    available_meetings = sorted(
        {str(d.metadata["meeting"]) for d in indexed_docs},
        key=lambda m: (0, int(m)) if m.isdigit() else (1, m),
    )
    print(f"Loaded {len(indexed_docs)} chunks across meetings: {available_meetings}")

    questions = load_or_init_questions(golden_path)
    if bootstrap:
        target = golden_path or DEFAULT_GOLDEN_PATH
        export_bootstrap_golden(questions, indexed_docs, target)
        return

    strategies = ["Dense MMR (bge-m3)", "Sparse BM25", "Hybrid RRF (Dense + BM25)"]
    metrics_by_strategy: dict[str, list[dict[str, float]]] = {s: [] for s in strategies}
    latencies_by_strategy: dict[str, list[float]] = {s: [] for s in strategies}
    per_query_log: list[dict] = []

    for item in questions:
        active_meetings = [m for m in item.meetings if m in available_meetings]
        if not active_meetings:
            print(f"[SKIP] {item.id}: none of {item.meetings} found in ChromaDB.")
            continue

        gold_keys = resolve_gold_chunks(item, indexed_docs)
        if not gold_keys:
            print(f"[SKIP] {item.id}: no chunks matched {item.evidence_patterns}.")
            continue

        scoped_docs = [
            d for d in indexed_docs if str(d.metadata["meeting"]) in active_meetings
        ]

        t0 = time.perf_counter()
        dense_docs = retrieve_dense(vector_store, item.question, active_meetings, k=6)
        lat_dense = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        bm25_docs = retrieve_bm25(scoped_docs, item.question, k=6)
        lat_bm25 = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        hybrid_docs = retrieve_hybrid(
            vector_store, scoped_docs, item.question, active_meetings, top_k=6
        )
        lat_hybrid = (time.perf_counter() - t0) * 1000

        runs = [
            ("Dense MMR (bge-m3)", deduplicate_keys(dense_docs), lat_dense),
            ("Sparse BM25", deduplicate_keys(bm25_docs), lat_bm25),
            ("Hybrid RRF (Dense + BM25)", deduplicate_keys(hybrid_docs), lat_hybrid),
        ]

        query_record: dict = {
            "id": item.id,
            "category": item.category,
            "question": item.question,
            "meetings": active_meetings,
            "gold_chunk_ids": gold_keys,
            "results": {},
        }

        for name, retrieved_keys, latency_ms in runs:
            scores = compute_query_metrics(retrieved_keys, gold_keys)
            metrics_by_strategy[name].append(scores)
            latencies_by_strategy[name].append(latency_ms)
            query_record["results"][name] = {
                "retrieved": retrieved_keys,
                "latency_ms": round(latency_ms, 2),
                **{k: round(v, 4) for k, v in scores.items()},
            }

        per_query_log.append(query_record)

    evaluated_count = len(per_query_log)
    if evaluated_count == 0:
        raise RuntimeError(
            "No evaluation queries could be scored against the indexed meetings."
        )

    summary_rows: list[dict] = []
    for name in strategies:
        rows = metrics_by_strategy[name]
        lats = latencies_by_strategy[name]
        summary_rows.append(
            {
                "retriever": name,
                "queries": evaluated_count,
                "recall@1": round(
                    sum(r["recall@1"] for r in rows) / evaluated_count, 4
                ),
                "recall@3": round(
                    sum(r["recall@3"] for r in rows) / evaluated_count, 4
                ),
                "recall@6": round(
                    sum(r["recall@6"] for r in rows) / evaluated_count, 4
                ),
                "hit@6": round(sum(r["hit@6"] for r in rows) / evaluated_count, 4),
                "mrr@6": round(sum(r["mrr@6"] for r in rows) / evaluated_count, 4),
                "avg_latency_ms": round(sum(lats) / evaluated_count, 1),
            }
        )

    print("\n### Retrieval Evaluation Summary (Markdown for README.md)\n")
    print(
        "| Retriever | Queries | Recall@1 | Recall@3 | Recall@6 | HitRate@6 | MRR@6 | Avg Latency |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        print(
            f"| {row['retriever']} | {row['queries']} | "
            f"{row['recall@1']:.3f} | {row['recall@3']:.3f} | {row['recall@6']:.3f} | "
            f"{row['hit@6']:.3f} | {row['mrr@6']:.3f} | {row['avg_latency_ms']:.1f} ms |"
        )

    if results_path is not None:
        results_path.write_text(
            json.dumps(
                {
                    "collection": COLLECTION_NAME,
                    "embedding_model": EMBEDDING_MODEL,
                    "indexed_meetings": available_meetings,
                    "total_chunks": len(indexed_docs),
                    "summary": summary_rows,
                    "queries": per_query_log,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved detailed benchmark report to {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Dense vs. BM25 vs. Hybrid RRF retrieval over Copom minutes."
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="Optional path to a golden evaluation JSON dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save detailed JSON benchmark results.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Export golden_copom_eval.json with resolved chunk IDs and snippet previews.",
    )
    args = parser.parse_args()
    run_benchmark(
        golden_path=args.golden, results_path=args.output, bootstrap=args.bootstrap
    )


if __name__ == "__main__":
    main()
