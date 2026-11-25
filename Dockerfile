FROM apache/airflow:3.4.0
USER airflow
RUN python -m venv /opt/airflow/dbt_venv \
    && /opt/airflow/dbt_venv/bin/pip install --no-cache-dir dbt-core==1.12.7 dbt-postgres==1.11.0