import requests
import pendulum
from decimal import Decimal

from requests.exceptions import HTTPError
from airflow.sdk.exceptions import AirflowFailException
from airflow.sdk import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook

USER_AGENT_HEADER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    + "AppleWebKit/537.36 (KHTML, like Gecko) "
    + "Chrome/111.0.0.0 Safari/537.36"
}

DBT_BIN = "/opt/airflow/dbt_venv/bin/dbt"

DBT_PROJECT_DIR = '/opt/airflow/dbt_project/bacen_analytics'


@dag(
    schedule="@daily",
    start_date=pendulum.datetime(2026, 11, 9, tz="UTC"),
    catchup=False,
    tags=["bacen"],
    max_active_runs=2,
)
def ingest_from_bacen():

    @task()
    def init_db():
        hook = PostgresHook(postgres_conn_id="postgres_default")
        hook.run("CREATE SCHEMA IF NOT EXISTS raw")
        hook.run(
            "CREATE TABLE IF NOT EXISTS raw.raw_bacen_selic ("
            "data DATE PRIMARY KEY, valor NUMERIC(6,2), ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW())"
        )

    @task(retries=3, retry_delay=pendulum.duration(seconds=10))
    def extract(
        data_interval_start=None,
        data_interval_end=None,
        logical_date=None,
        dag_run=None,
    ):
        conf = dag_run.conf if dag_run and dag_run.conf else {}
        if conf.get("data_inicial") and conf.get("data_final"):
            start_str = conf.get("data_inicial")
            end_str = conf.get("data_final")
        else:
            if data_interval_start is None:
                data_interval_start = logical_date
            if data_interval_end is None:
                data_interval_end = logical_date

            start_str = data_interval_start.format("DD/MM/YYYY")
            end_str = data_interval_end.format("DD/MM/YYYY")

        payload = {"formato": "json", "dataInicial": start_str, "dataFinal": end_str}

        try:
            r = requests.get(
                url="https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados",
                params=payload,
                headers=USER_AGENT_HEADER,
                timeout=10,
            )
            r.raise_for_status()
            return r.json()
        except HTTPError as err:
            if err.response.status_code == 404:
                raise AirflowFailException("Error 404: Check the URL")
            else:
                raise

    @task(retries=3, retry_delay=pendulum.duration(seconds=10))
    def load(records):
        insertion_list = [
            (
                pendulum.from_format(item["data"], "DD/MM/YYYY").to_date_string(),
                Decimal(item["valor"]),
            )
            for item in records
        ]
        hook = PostgresHook(postgres_conn_id="postgres_default")
        with hook.get_conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO raw.raw_bacen_selic (data, valor) "
                    "VALUES (%s, %s) "
                    "ON CONFLICT (data) DO UPDATE "
                    "SET valor = EXCLUDED.valor, ingested_at = NOW()",
                    insertion_list,
                )

    @task.bash(max_active_tis_per_dag=1)
    def build_dbt() -> str:
        return f"{DBT_BIN} build --project-dir {DBT_PROJECT_DIR} --profiles-dir {DBT_PROJECT_DIR}"

    init_db() >> load(extract()) >> build_dbt()


ingest_from_bacen()
