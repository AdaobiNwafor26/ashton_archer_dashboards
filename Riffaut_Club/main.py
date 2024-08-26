import streamlit as st
from utils.data_loader import load_data
from datetime import date
import pandas as pd

# st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")

# # Custom CSS for the blue-dark blue theme
# st.markdown("""
# <style>
#     .reportview-container {
#         background: linear-gradient(to right, #1e3c72, #2a5298);
#     }
#     .sidebar .sidebar-content {
#         background: linear-gradient(to bottom, #2a5298, #1e3c72);
#     }
#     .Widget>label {
#         color: white;
#     }
#     .stplot {
#         background-color: rgba(255, 255, 255, 0.1);
#     }
# </style>
# """, unsafe_allow_html=True)

# # Load your data
# (df, df_customers_raw, df_orders_raw, df_transactions_raw, monthly_revenue, quarterly_revenue, customer_reten_monthly, 
#             customer_reten_quarterly, churn_rate_monthly, churn_rate_quarterly, cltv, customer_churn_probability, 
#             rfm_table, rfm_customer_segment, rfm_segment_final, product_affinity) = load_data()


# # Sidebar filters
# st.sidebar.header("Filters")
# df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
# start_date = st.sidebar.date_input("Start Date", df['Invoice Date'].min())
# end_date = st.sidebar.date_input("End Date", df['Invoice Date'].max())

# start_date = pd.to_datetime(start_date)
# end_date = pd.to_datetime(end_date)

# df_filtered = df[(df['Invoice Date'] >= start_date) & (df['Invoice Date'] <= end_date)]

# st.session_state['df_filtered'] = df_filtered
# st.session_state['start_date'] = start_date
# st.session_state['end_date'] = end_date
# st.session_state['churn_rate_monthly'] = churn_rate_monthly


