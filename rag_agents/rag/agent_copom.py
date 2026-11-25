import os
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
import psycopg
from psycopg.rows import dict_row

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.errors import GraphRecursionError

from rag.chain_copom import chat_model, query_copom, vector_store

project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")
load_dotenv()


SYSTEM_PROMPT = (
    "Você é um analista macroeconômico e de dados financeiros especializado na taxa SELIC "
    "e nas atas do Comitê de Política Monetária (Copom) do Banco Central do Brasil (BACEN). "
    "Responda sempre em português claro, objetivo e estritamente fundamentado nos dados retornados pelas ferramentas.\n\n"
    "Regras obrigatórias de roteamento e fundamentação:\n"
    "1. Dados quantitativos da taxa SELIC (série histórica):\n"
    "   - Utilize `get_selic_monthly_summary` para tendências mensais, anuais, médias, mínimas, máximas ou taxas de fechamento.\n"
    "   - Utilize `get_selic_daily_rates` apenas quando o usuário pedir taxas diárias ou datas específicas.\n"
    "   - Nunca invente, estime ou arredonde valores fora dos registros retornados pelo banco de dados.\n"
    "2. Dados qualitativos das Atas do Copom:\n"
    "   - Se o usuário não informar explicitamente os números das reuniões, chame primeiro `list_available_copom_meetings` para descobrir as atas disponíveis.\n"
    "   - Em seguida, invoque imediatamente a ferramenta `search_copom_minutes` (via tool_call estruturado — NUNCA escreva chamadas de função como texto na resposta nem peça para o usuário aguardar).\n"
    "   - Passe todos os IDs de reuniões relevantes do período em uma ÚNICA chamada (ex.: `meetings=['276', '277', '278', '279']`), em vez de fazer uma chamada por reunião.\n"
    "   - Ao apresentar justificativas, projeções ou decisões do comitê, cite explicitamente o número da reunião (ex.: Reunião 279).\n"
    "3. Perguntas híbridas (números da SELIC + justificativa do Copom):\n"
    "   - Sempre que a pergunta envolver números da SELIC, chame a ferramenta SQL correspondente imediatamente na primeira etapa, independentemente das atas disponíveis.\n"
    "   - Nunca afirme que consultou uma ferramenta ou que não há dados no banco sem antes ter executado a chamada dessa ferramenta."
    "4. Limite de abstenção (anti-loop):\n"
    "   - Se uma consulta retornar lista vazia ou se `search_copom_minutes` retornar `has_sufficient_context=False`, não repita a mesma chamada em loop. "
    "Informe claramente ao usuário quais dados ou períodos não estão disponíveis na base."
    "5. Alinhamento temporal entre fontes (anti-alucinação):\n"
    "   - Verifique sempre se o ano/período das atas retornadas por `list_available_copom_meetings` corresponde ao período perguntado pelo usuário.\n"
    "   - Ao chamar `search_copom_minutes`, mantenha explicitamente o ano ou período na pergunta (ex.: 'justificativa do Copom no início de 2025').\n"
    "   - Nunca atribua justificativas de atas de um ano (ex.: 2026) para explicar taxas SELIC de outro ano (ex.: 2025). "
    "Se as atas disponíveis forem de outro período, informe claramente essa limitação temporal ao usuário."
)


def agent_node(state: MessagesState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def ask_agent(question: str) -> None:
    print("=" * 70)
    print("Question:", question)
    t0 = time.perf_counter()
    try:
        result = agent_graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": 10},
        )
    except GraphRecursionError:
        print(f"Elapsed: {time.perf_counter() - t0:.2f}s")
        print("Final response: Limite de iterações atingido sem resposta conclusiva.")
        return

    tool_trajectory = [
        {"name": tc["name"], "args": tc["args"]}
        for msg in result["messages"]
        for tc in getattr(msg, "tool_calls", [])
    ]
    print(f"Elapsed: {time.perf_counter() - t0:.2f}s")
    print("Tool trajectory:", tool_trajectory)
    print("Final response:\n", result["messages"][-1].content)


def get_readonly_conn() -> psycopg.Connection:
    return psycopg.connect(
        host="localhost",
        port=5432,
        user=os.environ["POSTGRES_USER"],
        dbname=os.environ["POSTGRES_DB"],
        password=os.environ["POSTGRES_PASSWORD"],
        connect_timeout=5,
        autocommit=True,
        options="-c default_transaction_read_only=on -c statement_timeout=5000",
    )


@tool
def get_selic_monthly_summary(start_date: str, end_date: str) -> list[dict]:
    """Fetch monthly aggregated SELIC interest rate statistics from the PostgreSQL analytics warehouse.

    Use this tool for multi-month or yearly SELIC rate trends, averages, minimums, maximums,
    or closing rates instead of daily observations.

    Args:
        start_date: Inclusive start date in ISO format 'YYYY-MM-DD' (e.g., '2025-01-01').
        end_date: Inclusive end date in ISO format 'YYYY-MM-DD' (e.g., '2025-12-31').
    """
    parsed_start = date.fromisoformat(start_date)
    parsed_end = date.fromisoformat(end_date)

    if parsed_start > parsed_end:
        raise ValueError("start_date must be before or equal to end_date.")

    sql = """
        SELECT
            month_start,
            avg_selic_rate,
            min_selic_rate,
            max_selic_rate,
            observations_count,
            closing_selic_rate
        FROM analytics.mart_selic_monthly_summary
        WHERE month_start BETWEEN %s AND %s
        ORDER BY month_start ASC
    """

    with get_readonly_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (parsed_start, parsed_end))
        rows = cur.fetchall()

    return [
        {
            "month_start": row["month_start"].isoformat(),
            "avg_selic_rate": float(row["avg_selic_rate"]),
            "min_selic_rate": float(row["min_selic_rate"]),
            "max_selic_rate": float(row["max_selic_rate"]),
            "observations_count": int(row["observations_count"]),
            "closing_selic_rate": float(row["closing_selic_rate"]),
        }
        for row in rows
    ]


@tool
def get_selic_daily_rates(start_date: str, end_date: str) -> list[dict]:
    """Fetch daily SELIC target interest rate observations from the PostgreSQL analytics warehouse.

    Use this tool only when exact daily SELIC rates or specific dates of rate changes are needed
    (capped at 366 daily rows per call). For multi-month or multi-year trends, use
    get_selic_monthly_summary instead.

    Args:
        start_date: Inclusive start date in ISO format 'YYYY-MM-DD' (e.g., '2025-01-01').
        end_date: Inclusive end date in ISO format 'YYYY-MM-DD' (e.g., '2025-01-31').
    """
    parsed_start = date.fromisoformat(start_date)
    parsed_end = date.fromisoformat(end_date)

    if parsed_start > parsed_end:
        raise ValueError("start_date must be before or equal to end_date.")

    sql = """
        SELECT
            reference_date,
            selic_rate
        FROM analytics.stg_bacen_selic
        WHERE reference_date BETWEEN %s AND %s
        ORDER BY reference_date ASC
        LIMIT 366
    """

    with get_readonly_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (parsed_start, parsed_end))
        rows = cur.fetchall()

    return [
        {
            "reference_date": row["reference_date"].isoformat(),
            "selic_rate": float(row["selic_rate"]),
        }
        for row in rows
    ]


@tool
def list_available_copom_meetings() -> list[dict]:
    """List all BACEN Copom meetings ('Atas do Copom') indexed in ChromaDB along with their opening header text.

    Always call this tool first before calling search_copom_minutes to inspect which meeting numbers
    and dates/years exist in the database.
    """
    raw = vector_store.get(include=["documents", "metadatas"])
    earliest_by_meeting: dict[str, tuple[int, str]] = {}

    for doc, meta in zip(raw.get("documents") or [], raw.get("metadatas") or []):
        if not meta or "meeting" not in meta:
            continue
        meeting_id = str(meta["meeting"])
        start_idx = int(meta.get("start_index", 0))
        if (
            meeting_id not in earliest_by_meeting
            or start_idx < earliest_by_meeting[meeting_id][0]
        ):
            preview = " ".join(doc.split())[:220]
            earliest_by_meeting[meeting_id] = (start_idx, preview)

    sorted_ids = sorted(
        earliest_by_meeting.keys(),
        key=lambda m: (0, int(m)) if m.isdigit() else (1, m),
    )
    return [
        {"meeting": m, "header_preview": earliest_by_meeting[m][1]} for m in sorted_ids
    ]


@tool
def search_copom_minutes(question: str, meetings: list[str]) -> dict:
    """Search BACEN Copom Meeting Minutes ('Atas do Copom') for qualitative policy rationale and decisions.

    Use this tool for Copom committee discussions, inflation outlook, balance of risks, and
    monetary policy justifications—never for raw historical SELIC time-series queries.

    Args:
        question: Specific question written in Portuguese matching the Copom minutes. MUST preserve
            any year, month, or time period requested by the user (e.g., 'no início de 2025' or 'em 2026').
        meetings: Non-empty list of valid Copom meeting ID strings (e.g., ['280', '281', '282'])
            obtained from list_available_copom_meetings.
    """
    if not meetings:
        raise ValueError(
            "meetings must be a non-empty list; call list_available_copom_meetings first."
        )
    result = query_copom(question=question, meetings=meetings)
    return result.model_dump()


TOOLS = [
    get_selic_monthly_summary,
    get_selic_daily_rates,
    list_available_copom_meetings,
    search_copom_minutes,
]

llm_with_tools = chat_model.bind_tools(TOOLS)

workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(TOOLS))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

agent_graph = workflow.compile()

if __name__ == "__main__":
    ask_agent(
        "Qual foi a taxa Selic média, mínima, máxima e de fechamento em cada mês do primeiro trimestre de 2025?"
    )
    ask_agent(
        "Quais foram os principais fatores de risco para a inflação destacados nas atas mais recentes do Copom?"
    )
    ask_agent(
        "Como evoluiu a taxa Selic no início de 2025 e qual foi a justificativa do Copom nas atas de 2025 para essa condução?"
    )
    ask_agent(
        "Como evoluiu a taxa Selic em 2026 e qual foi a justificativa do Copom nas atas de 2026 para a condução da política monetária?"
    )
    """
    meetings = list_available_copom_meetings.invoke({})
    print("Available meetings:", meetings)

    monthly = get_selic_monthly_summary.invoke(
        {"start_date": "2025-01-01", "end_date": "2025-03-31"}
    )
    print("Monthly rows:", len(monthly), monthly[:1])

    daily = get_selic_daily_rates.invoke(
        {"start_date": "2025-01-02", "end_date": "2025-01-10"}
    )
    print("Daily rows:", len(daily), daily[:2])

    rag_out = search_copom_minutes.invoke(
        {
            "question": "Qual foi a decisão sobre a taxa Selic?",
            "meetings": meetings[:2] if meetings else ["280"],
        }
    )
    print(
        "RAG sufficient:",
        rag_out["has_sufficient_context"],
        "Citations:",
        rag_out["citations"],
    )

    ai_msg = llm_with_tools.invoke("Quais atas do Copom estão disponíveis na base?")
    print("LLM tool_calls:", ai_msg.tool_calls)
    """
