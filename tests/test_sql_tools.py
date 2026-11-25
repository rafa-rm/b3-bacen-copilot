from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock
import pytest

import rag.agent_copom as agent_mod
from rag.agent_copom import (
    get_readonly_conn,
    get_selic_daily_rates,
    get_selic_monthly_summary,
)


def _mock_db_connection(monkeypatch: pytest.MonkeyPatch, rows: list[dict]) -> MagicMock:
    """Stub get_readonly_conn() context manager and return the mock cursor."""
    mock_cursor = MagicMock(name="MockCursor")
    mock_cursor.fetchall.return_value = rows

    cursor_cm = MagicMock()
    cursor_cm.__enter__.return_value = mock_cursor
    cursor_cm.__exit__.return_value = False

    mock_conn = MagicMock(name="MockConnection")
    mock_conn.cursor.return_value = cursor_cm

    conn_cm = MagicMock()
    conn_cm.__enter__.return_value = mock_conn
    conn_cm.__exit__.return_value = False

    mock_factory = MagicMock(return_value=conn_cm)
    monkeypatch.setattr(agent_mod, "get_readonly_conn", mock_factory)
    return mock_cursor


class TestReadonlyConnection:
    def test_get_readonly_conn_enforces_readonly_and_statement_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("POSTGRES_USER", "readonly_tester")
        monkeypatch.setenv("POSTGRES_DB", "bacen_db")
        monkeypatch.setenv("POSTGRES_PASSWORD", "secret_pass")

        mock_connect = MagicMock(name="psycopg_connect")
        monkeypatch.setattr(agent_mod.psycopg, "connect", mock_connect)

        get_readonly_conn()

        mock_connect.assert_called_once()
        kwargs = mock_connect.call_args.kwargs
        assert kwargs["host"] == "localhost"
        assert kwargs["port"] == 5432
        assert kwargs["user"] == "readonly_tester"
        assert kwargs["dbname"] == "bacen_db"
        assert kwargs["password"] == "secret_pass"
        assert kwargs["autocommit"] is True
        assert kwargs["connect_timeout"] == 5
        assert "default_transaction_read_only=on" in kwargs["options"]
        assert "statement_timeout=5000" in kwargs["options"]


class TestGetSelicMonthlySummary:
    def test_valid_range_executes_parameterized_query_and_serializes_types(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sample_rows = [
            {
                "month_start": date(2025, 1, 1),
                "avg_selic_rate": Decimal("12.35"),
                "min_selic_rate": Decimal("12.25"),
                "max_selic_rate": Decimal("13.25"),
                "observations_count": 22,
                "closing_selic_rate": Decimal("13.25"),
            },
            {
                "month_start": date(2025, 2, 1),
                "avg_selic_rate": Decimal("13.25"),
                "min_selic_rate": Decimal("13.25"),
                "max_selic_rate": Decimal("13.25"),
                "observations_count": 20,
                "closing_selic_rate": Decimal("13.25"),
            },
        ]
        mock_cursor = _mock_db_connection(monkeypatch, sample_rows)

        result = get_selic_monthly_summary.invoke(
            {"start_date": "2025-01-01", "end_date": "2025-02-28"}
        )

        mock_cursor.execute.assert_called_once()
        sql_query, params = mock_cursor.execute.call_args.args
        assert "FROM analytics.mart_selic_monthly_summary" in sql_query
        assert "WHERE month_start BETWEEN %s AND %s" in sql_query
        assert "ORDER BY month_start ASC" in sql_query
        assert params == (date(2025, 1, 1), date(2025, 2, 28))

        assert result == [
            {
                "month_start": "2025-01-01",
                "avg_selic_rate": 12.35,
                "min_selic_rate": 12.25,
                "max_selic_rate": 13.25,
                "observations_count": 22,
                "closing_selic_rate": 13.25,
            },
            {
                "month_start": "2025-02-01",
                "avg_selic_rate": 13.25,
                "min_selic_rate": 13.25,
                "max_selic_rate": 13.25,
                "observations_count": 20,
                "closing_selic_rate": 13.25,
            },
        ]
        assert isinstance(result[0]["avg_selic_rate"], float)
        assert isinstance(result[0]["observations_count"], int)

    def test_allows_equal_start_and_end_dates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_cursor = _mock_db_connection(monkeypatch, [])
        result = get_selic_monthly_summary.invoke(
            {"start_date": "2025-03-01", "end_date": "2025-03-01"}
        )
        assert result == []
        _, params = mock_cursor.execute.call_args.args
        assert params == (date(2025, 3, 1), date(2025, 3, 1))

    @pytest.mark.parametrize(
        ("start_date", "end_date"),
        [
            ("01/01/2025", "2025-03-31"),
            ("2025-01-01", "31-03-2025"),
            ("2025-13-01", "2025-12-31"),
            ("invalid-date", "2025-03-31"),
        ],
    )
    def test_rejects_non_iso_dates_without_hitting_db(
        self, monkeypatch: pytest.MonkeyPatch, start_date: str, end_date: str
    ) -> None:
        conn_spy = MagicMock(side_effect=AssertionError("DB should not be called"))
        monkeypatch.setattr(agent_mod, "get_readonly_conn", conn_spy)

        with pytest.raises(ValueError):
            get_selic_monthly_summary.invoke(
                {"start_date": start_date, "end_date": end_date}
            )
        conn_spy.assert_not_called()

    def test_rejects_start_date_after_end_date_without_hitting_db(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn_spy = MagicMock(side_effect=AssertionError("DB should not be called"))
        monkeypatch.setattr(agent_mod, "get_readonly_conn", conn_spy)

        with pytest.raises(
            ValueError, match="start_date must be before or equal to end_date"
        ):
            get_selic_monthly_summary.invoke(
                {"start_date": "2025-06-01", "end_date": "2025-01-01"}
            )
        conn_spy.assert_not_called()


class TestGetSelicDailyRates:
    def test_valid_range_queries_staging_with_limit_366(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sample_rows = [
            {"reference_date": date(2025, 1, 2), "selic_rate": Decimal("12.25")},
            {"reference_date": date(2025, 1, 3), "selic_rate": Decimal("12.25")},
        ]
        mock_cursor = _mock_db_connection(monkeypatch, sample_rows)

        result = get_selic_daily_rates.invoke(
            {"start_date": "2025-01-02", "end_date": "2025-01-03"}
        )

        sql_query, params = mock_cursor.execute.call_args.args
        assert "FROM analytics.stg_bacen_selic" in sql_query
        assert "WHERE reference_date BETWEEN %s AND %s" in sql_query
        assert "ORDER BY reference_date ASC" in sql_query
        assert "LIMIT 366" in sql_query
        assert params == (date(2025, 1, 2), date(2025, 1, 3))
        assert result == [
            {"reference_date": "2025-01-02", "selic_rate": 12.25},
            {"reference_date": "2025-01-03", "selic_rate": 12.25},
        ]

    def test_rejects_start_date_after_end_date_without_hitting_db(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn_spy = MagicMock(side_effect=AssertionError("DB should not be called"))
        monkeypatch.setattr(agent_mod, "get_readonly_conn", conn_spy)

        with pytest.raises(
            ValueError, match="start_date must be before or equal to end_date"
        ):
            get_selic_daily_rates.invoke(
                {"start_date": "2025-02-10", "end_date": "2025-02-01"}
            )
        conn_spy.assert_not_called()
