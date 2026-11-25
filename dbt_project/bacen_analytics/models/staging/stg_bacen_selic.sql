with 

source as (
    select * from {{ source('raw', 'raw_bacen_selic')}}
),

renamed as (
    select data::date as reference_date, valor::numeric(6,2) as selic_rate, ingested_at::timestamptz 
    from source
)

select * from renamed