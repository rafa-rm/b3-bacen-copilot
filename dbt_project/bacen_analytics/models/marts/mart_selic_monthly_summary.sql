with 

staging as (
    select * from {{ ref('stg_bacen_selic')}}
),

monthly_aggregated as (
    select 
        DATE_TRUNC('month', reference_date)::date as month_start,
        ROUND(AVG(selic_rate), 2) as avg_selic_rate,
        MIN(selic_rate) as min_selic_rate,
        MAX(selic_rate) as max_selic_rate,
        COUNT(*) as observations_count,
        (ARRAY_AGG(selic_rate ORDER BY reference_date DESC))[1] as closing_selic_rate
    from staging
    group by 1
)

select * from monthly_aggregated