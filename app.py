import ast
import json
import time
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from rag.config import COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL

SAMPLE_QUERIES = [
    "Qual foi a taxa Selic média, mínima, máxima e de fechamento em cada mês do primeiro trimestre de 2025?",
    "Quais foram os principais fatores de risco para a inflação destacados nas atas mais recentes do Copom?",
    "Como evoluiu a taxa Selic no início de 2025 e qual foi a justificativa do Copom nas atas de 2025 para essa condução?",
    "Como evoluiu a taxa Selic em 2026 e qual foi a justificativa do Copom nas atas de 2026 para a condução da política monetária?",
]


@st.cache_resource(
    show_spinner="Inicializando modelo de embeddings (BAAI/bge-m3), ChromaDB e agente LangGraph (pode levar alguns segundos na primeira execução)..."
)
def _load_agent_graph():
    from rag.agent_copom import agent_graph

    return agent_graph


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        return "\n".join(p for p in parts if p)
    return str(content)


def _parse_tool_payload(raw_content: Any) -> Any:
    if not isinstance(raw_content, str):
        return raw_content
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw_content)
        except (ValueError, SyntaxError):
            return raw_content


def _run_agent(graph: Any, chat_history: list[dict[str, Any]]) -> dict[str, Any]:
    lc_messages = [
        (
            HumanMessage(content=msg["content"])
            if msg["role"] == "user"
            else AIMessage(content=msg["content"])
        )
        for msg in chat_history
    ]

    t0 = time.perf_counter()
    try:
        result = graph.invoke(
            {"messages": lc_messages},
            config={"recursion_limit": 10},
        )
    except GraphRecursionError:
        return {
            "content": "Limite de iterações atingido sem resposta conclusiva.",
            "elapsed": time.perf_counter() - t0,
            "trajectory": [],
            "rag_outputs": [],
            "sql_outputs": [],
        }

    elapsed = time.perf_counter() - t0
    all_messages = result["messages"]
    new_messages = all_messages[len(lc_messages) :]

    trajectory: list[dict[str, Any]] = []
    rag_outputs: list[dict[str, Any]] = []
    sql_outputs: list[dict[str, Any]] = []

    for msg in new_messages:
        for tc in getattr(msg, "tool_calls", []) or []:
            trajectory.append({"name": tc["name"], "args": tc["args"]})

        if isinstance(msg, ToolMessage):
            parsed = _parse_tool_payload(msg.content)
            if msg.name == "search_copom_minutes" and isinstance(parsed, dict):
                rag_outputs.append(parsed)
            elif msg.name in {
                "get_selic_monthly_summary",
                "get_selic_daily_rates",
            } and isinstance(parsed, list):
                sql_outputs.append({"tool": msg.name, "rows": parsed})

    final_content = _extract_text(new_messages[-1].content) if new_messages else ""
    return {
        "content": final_content,
        "elapsed": elapsed,
        "trajectory": trajectory,
        "rag_outputs": rag_outputs,
        "sql_outputs": sql_outputs,
    }


def _render_trace(trace: dict[str, Any]) -> None:
    trajectory = trace.get("trajectory", [])
    rag_outputs = trace.get("rag_outputs", [])
    sql_outputs = trace.get("sql_outputs", [])
    elapsed = trace.get("elapsed")

    if not (trajectory or rag_outputs or sql_outputs or elapsed is not None):
        return

    tool_names = " → ".join(step["name"] for step in trajectory) or "Resposta direta"
    header = f"Execução & Guardrails ({elapsed:.2f}s) | {tool_names}"

    with st.expander(header, expanded=False):
        if trajectory:
            st.markdown("**Trajetória de Ferramentas (LangGraph)**")
            for idx, step in enumerate(trajectory, start=1):
                st.code(
                    f"{idx}. {step['name']}({json.dumps(step['args'], ensure_ascii=False)})",
                    language="text",
                )

        for rag in rag_outputs:
            sufficient = rag.get("has_sufficient_context")
            status_label = (
                "Contexto suficiente"
                if sufficient
                else "Abstenção acionada (contexto insuficiente)"
            )
            st.markdown(f"**Guardrail RAG:** `{status_label}`")
            citations = rag.get("citations") or []
            if citations:
                st.markdown("**Citações verificadas:**")
                for citation in citations:
                    st.markdown(f"- {citation}")

        for sql in sql_outputs:
            rows = sql.get("rows") or []
            st.markdown(
                f"**Dados PostgreSQL (`{sql['tool']}` — {len(rows)} registros)**"
            )
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="B3 / BACEN Copilot",
        page_icon="📊",
        layout="wide",
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.title("B3 / BACEN Copilot")
        st.caption("SELIC Data Warehouse + Hybrid RRF Copom RAG")

        st.divider()
        st.markdown("**Stack Ativa**")
        st.markdown(
            f"- **LLM:** `{LLM_MODEL}`\n"
            f"- **Embeddings:** `{EMBEDDING_MODEL}`\n"
            f"- **Chroma Collection:** `{COLLECTION_NAME}`\n"
            f"- **Warehouse:** `PostgreSQL 18 / dbt Core`"
        )

        st.divider()
        st.markdown("**Perguntas de Exemplo**")
        for idx, sample in enumerate(SAMPLE_QUERIES):
            if st.button(sample, key=f"sample_{idx}", width="stretch"):
                st.session_state.pending_prompt = sample

        st.divider()
        if st.button("Limpar conversa", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    st.header("Copiloto Analítico — SELIC & Atas do Copom")
    st.markdown(
        "Consulte séries históricas da taxa **SELIC** (PostgreSQL/dbt) e decisões/riscos das **Atas do Copom** (ChromaDB Hybrid RAG)."
    )

    graph = _load_agent_graph()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "trace" in msg:
                _render_trace(msg["trace"])

    user_input = st.chat_input(
        "Pergunte sobre a taxa SELIC ou sobre as atas do Copom..."
    )
    prompt = st.session_state.pop("pending_prompt", None) or user_input

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consultando warehouse e atas do Copom..."):
                result = _run_agent(graph, st.session_state.messages)

            st.markdown(result["content"])
            trace = {
                "elapsed": result["elapsed"],
                "trajectory": result["trajectory"],
                "rag_outputs": result["rag_outputs"],
                "sql_outputs": result["sql_outputs"],
            }
            _render_trace(trace)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["content"],
                "trace": trace,
            }
        )


if __name__ == "__main__":
    main()
