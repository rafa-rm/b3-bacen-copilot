import requests
import pendulum

from requests.exceptions import HTTPError
from airflow.sdk.exceptions import AirflowFailException
from airflow.sdk import dag, task

USER_AGENT_HEADER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    + "AppleWebKit/537.36 (KHTML, like Gecko) "
    + "Chrome/111.0.0.0 Safari/537.36"
}


@dag(
    schedule=None,
    start_date=pendulum.datetime(2026, 11, 9, tz="UTC"),
    catchup=False,
    tags=["bacen"],
)
def ingest_from_bacen():

    @task(retries=3, retry_delay=pendulum.duration(seconds=10))
    def extract():
        try:
            r = requests.get(
                url="https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/"
                + "dados/ultimos/10?formato=json",
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

    extract()


ingest_from_bacen()
