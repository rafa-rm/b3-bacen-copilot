from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

import rag.agent_copom as agent_mod
import rag.chain_copom as chain_mod
from rag.agent_copom import (
    agent_graph,
    ask_agent,
    list_available_copom_meetings,
    search_copom_minutes,
)
from rag.chain_copom import ABSTENTION_MSG, CopomResponse, query_copom


class TestCopomToolWrappers:
    def test_list_available_copom_meetings_picks_earliest_chunk_and_sorts_numerically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_store = MagicMock()
        mock_store.get.return_value = {
            "documents": [
                "Trecho intermediário da reunião 282",
                "   Notas da 282ª Reunião   do Copom - 18 de março de 2026   ",
                "Notas da 276ª Reunião do Copom - 30 de julho de 2025",
                "Notas da 280ª Reunião do Copom - 10 de dezembro de 2025",
            ],
            "metadatas": [
                {"meeting": "282", "start_index": 2400},
                {"meeting": "282", "start_index": 0},
                {"meeting": "276", "start_index": 0},
                {"meeting": "280", "start_index": 0},
            ],
        }
        monkeypatch.setattr(agent_mod, "vector_store", mock_store)

        meetings = list_available_copom_meetings.invoke({})

        assert [m["meeting"] for m in meetings] == ["276", "280", "282"]
        assert meetings[2]["header_preview"] == (
            "Notas da 282ª Reunião do Copom - 18 de março de 2026"
        )

    def test_search_copom_minutes_rejects_empty_meetings_list(self) -> None:
        with pytest.raises(ValueError, match="meetings must be a non-empty list"):
            search_copom_minutes.invoke(
                {"question": "Qual foi a decisão da Selic?", "meetings": []}
            )

    def test_search_copom_minutes_delegates_to_query_copom(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = CopomResponse(
            answer="Na Reunião 282, o Copom manteve a taxa Selic em 14,25% a.a.",
            citations=[1200],
            has_sufficient_context=True,
        )
        mock_query = MagicMock(return_value=expected)
        monkeypatch.setattr(agent_mod, "query_copom", mock_query)

        output = search_copom_minutes.invoke(
            {"question": "Qual a decisão na reunião 282?", "meetings": ["282"]}
        )

        mock_query.assert_called_once_with(
            question="Qual a decisão na reunião 282?", meetings=["282"]
        )
        assert output == expected.model_dump()


class TestCitationGroundingAndAbstention:
    def test_filters_out_fabricated_citations_while_keeping_valid_ones(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_chain = RunnableLambda(
            lambda _: {
                "context": [
                    Document(
                        page_content="Trecho A",
                        metadata={"meeting": "282", "start_index": 100},
                    ),
                    Document(
                        page_content="Trecho B",
                        metadata={"meeting": "282", "start_index": 900},
                    ),
                ],
                "structured_response": CopomResponse(
                    answer="O Copom destacou riscos fiscais na Reunião 282.",
                    citations=[100, 99999],
                    has_sufficient_context=True,
                ),
            }
        )
        monkeypatch.setattr(chain_mod, "rag_chain", fake_chain)

        res = query_copom("Quais os riscos fiscais?", ["282"])
        assert res.has_sufficient_context is True
        assert res.citations == [100]

    def test_downgrades_to_abstention_when_all_citations_are_fabricated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_chain = RunnableLambda(
            lambda _: {
                "context": [
                    Document(
                        page_content="Trecho A",
                        metadata={"meeting": "282", "start_index": 100},
                    ),
                ],
                "structured_response": CopomResponse(
                    answer="Resposta inventada.",
                    citations=[77777],
                    has_sufficient_context=True,
                ),
            }
        )
        monkeypatch.setattr(chain_mod, "rag_chain", fake_chain)

        res = query_copom("Pergunta fora do escopo", ["282"])
        assert res.has_sufficient_context is False
        assert res.citations == []
        assert res.answer == ABSTENTION_MSG

    def test_recovers_inline_citations_when_citations_field_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_chain = RunnableLambda(
            lambda _: {
                "context": [
                    Document(
                        page_content="Trecho A",
                        metadata={"meeting": "281", "start_index": 450},
                    ),
                ],
                "structured_response": CopomResponse(
                    answer="Conforme [start_index: 450], a inflação segue acima da meta.",
                    citations=[],
                    has_sufficient_context=True,
                ),
            }
        )
        monkeypatch.setattr(chain_mod, "rag_chain", fake_chain)

        res = query_copom("Como está a inflação?", ["281"])
        assert res.has_sufficient_context is True
        assert res.citations == [450]


class TestLangGraphAgentRouting:
    def test_routes_quantitative_selic_query_to_monthly_sql_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_monthly = MagicMock(
            return_value=[{"month_start": "2025-01-01", "closing_selic_rate": 13.25}]
        )
        monkeypatch.setattr(agent_mod.get_selic_monthly_summary, "func", mock_monthly)

        def scripted_llm(messages: list) -> AIMessage:
            tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
            if not tool_msgs:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_selic_monthly_summary",
                            "args": {
                                "start_date": "2025-01-01",
                                "end_date": "2025-03-31",
                            },
                            "id": "call_sql_1",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(
                content="Em janeiro de 2025, a taxa Selic de fechamento foi de 13,25% a.a."
            )

        monkeypatch.setattr(agent_mod, "llm_with_tools", RunnableLambda(scripted_llm))

        result = agent_graph.invoke(
            {
                "messages": [
                    HumanMessage(content="Qual foi a Selic no 1º trimestre de 2025?")
                ]
            }
        )

        mock_monthly.assert_called_once_with(
            start_date="2025-01-01", end_date="2025-03-31"
        )
        executed_tools = [
            tc["name"]
            for msg in result["messages"]
            for tc in getattr(msg, "tool_calls", [])
        ]
        assert executed_tools == ["get_selic_monthly_summary"]
        assert "13,25%" in result["messages"][-1].content

    def test_routes_qualitative_copom_query_through_discovery_then_rag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            agent_mod.list_available_copom_meetings,
            "func",
            MagicMock(
                return_value=[
                    {
                        "meeting": "281",
                        "header_preview": "Notas da 281ª Reunião - Jan 2026",
                    },
                    {
                        "meeting": "282",
                        "header_preview": "Notas da 282ª Reunião - Mar 2026",
                    },
                ]
            ),
        )
        mock_search = MagicMock(
            return_value={
                "answer": "Na Reunião 282, o Copom apontou desancoragem das expectativas.",
                "citations": [3100],
                "has_sufficient_context": True,
            }
        )
        monkeypatch.setattr(agent_mod.search_copom_minutes, "func", mock_search)

        def scripted_llm(messages: list) -> AIMessage:
            tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
            if len(tool_msgs) == 0:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "list_available_copom_meetings",
                            "args": {},
                            "id": "call_list_1",
                            "type": "tool_call",
                        }
                    ],
                )
            if len(tool_msgs) == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_copom_minutes",
                            "args": {
                                "question": "Quais os riscos de inflação nas reuniões 281 e 282?",
                                "meetings": ["281", "282"],
                            },
                            "id": "call_rag_1",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(
                content="Segundo a Reunião 282, o principal risco é a desancoragem das expectativas."
            )

        monkeypatch.setattr(agent_mod, "llm_with_tools", RunnableLambda(scripted_llm))

        result = agent_graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Quais os riscos de inflação nas atas recentes?"
                    )
                ]
            }
        )

        mock_search.assert_called_once()
        executed_tools = [
            tc["name"]
            for msg in result["messages"]
            for tc in getattr(msg, "tool_calls", [])
        ]
        assert executed_tools == [
            "list_available_copom_meetings",
            "search_copom_minutes",
        ]

    def test_hybrid_temporal_mismatch_abstains_without_searching_wrong_year_minutes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When user asks about 2025 minutes but only 2026 meetings exist, agent reports SQL data
        and abstains from explaining 2025 policy using 2026 Copom minutes."""
        monkeypatch.setattr(
            agent_mod.get_selic_monthly_summary,
            "func",
            MagicMock(
                return_value=[
                    {"month_start": "2025-01-01", "closing_selic_rate": 13.25}
                ]
            ),
        )
        monkeypatch.setattr(
            agent_mod.list_available_copom_meetings,
            "func",
            MagicMock(
                return_value=[
                    {
                        "meeting": "282",
                        "header_preview": "Notas da 282ª Reunião - 18 de março de 2026",
                    }
                ]
            ),
        )
        mock_search = MagicMock(
            side_effect=AssertionError(
                "Should not search 2026 minutes for a 2025 question"
            )
        )
        monkeypatch.setattr(agent_mod.search_copom_minutes, "func", mock_search)

        def scripted_llm(messages: list) -> AIMessage:
            tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
            if not tool_msgs:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_selic_monthly_summary",
                            "args": {
                                "start_date": "2025-01-01",
                                "end_date": "2025-03-31",
                            },
                            "id": "call_sql_2025",
                            "type": "tool_call",
                        },
                        {
                            "name": "list_available_copom_meetings",
                            "args": {},
                            "id": "call_list_2025",
                            "type": "tool_call",
                        },
                    ],
                )
            return AIMessage(
                content=(
                    "A Selic fechou janeiro de 2025 em 13,25% a.a., porém a base de atas do Copom "
                    "contém apenas reuniões de 2026 (Reunião 282), não sendo possível apresentar "
                    "as justificativas das atas de 2025."
                )
            )

        monkeypatch.setattr(agent_mod, "llm_with_tools", RunnableLambda(scripted_llm))

        result = agent_graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Como evoluiu a Selic no início de 2025 e qual a justificativa nas atas de 2025?"
                    )
                ]
            }
        )

        mock_search.assert_not_called()
        assert "apenas reuniões de 2026" in result["messages"][-1].content

    def test_ask_agent_handles_recursion_limit_gracefully(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            agent_mod.list_available_copom_meetings,
            "func",
            MagicMock(return_value=[]),
        )
        infinite_loop_llm = RunnableLambda(
            lambda _: AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_available_copom_meetings",
                        "args": {},
                        "id": "loop_call",
                        "type": "tool_call",
                    }
                ],
            )
        )
        monkeypatch.setattr(agent_mod, "llm_with_tools", infinite_loop_llm)

        ask_agent("Pergunta que causa loop")
        captured = capsys.readouterr().out
        assert "Limite de iterações atingido sem resposta conclusiva." in captured
