import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_excel('notebooks/v2/df_customers_full.xlsx')
    df_customers_raw = pd.read_excel('notebooks/v2/df_customers.xlsx')
    df_orders_raw = pd.read_excel('notebooks/v2/df_orders.xlsx')
    df_transactions_raw = pd.read_excel('notebooks/v2/df_transactions.xlsx')
    monthly_revenue = pd.read_excel('notebooks/v2/monthly_revenue_trends.xlsx')
    quarterly_revenue = pd.read_excel('notebooks/v2/quarterly_revenue_trends.xlsx')
    customer_reten_monthly = pd.read_excel('notebooks/v2/df_customer_reten_monthly.xlsx')
    customer_reten_quarterly = pd.read_excel('notebooks/v2/df_customer_reten_quarterly.xlsx')
    churn_rate_monthly = pd.read_excel('notebooks/v2/customer_monthly_churn_rate.xlsx')
    churn_rate_quarterly = pd.read_excel('notebooks/v2/customer_quarterly_churn_rate.xlsx')
    cltv = pd.read_excel('notebooks/v2/customer_lifetime_value.xlsx')
    customer_churn_probability = pd.read_excel('notebooks/v2/churn_probability_by_customer.xlsx')
    rfm_table = pd.read_excel('notebooks/v2/rfm_table.xlsx')
    rfm_customer_segment = pd.read_excel('notebooks/v2/rfm_customer_segment.xlsx')
    rfm_segment_final = pd.read_excel('notebooks/v2/rfm_segment_final.xlsx')
    product_affinity = pd.read_excel('notebooks/v2/productivity_affinity.xlsx')

    return (df, df_customers_raw, df_orders_raw, df_transactions_raw, monthly_revenue, quarterly_revenue, customer_reten_monthly, 
            customer_reten_quarterly, churn_rate_monthly, churn_rate_quarterly, cltv, customer_churn_probability, 
            rfm_table, rfm_customer_segment, rfm_segment_final, product_affinity)
