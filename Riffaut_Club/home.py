import streamlit as st
import pandas as pd
from datetime import date
from utils.data_loader import load_data
import plotly.graph_objects as go
# from main import *


st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")

# Custom CSS for the blue-dark blue theme
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(to right, #1e3c72, #2a5298);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #2a5298, #1e3c72);
    }
    .Widget>label {
        color: white;
    }
    .stplot {
        background-color: rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# import streamlit_option_menu as option_menu

# selected = option_menu(
#     menu_title=None,  # required
#     options=["Home", "Settings"],  # required
#     icons=["house", "gear"],  # optional
#     menu_icon="cast",  # optional
#     default_index=0,  # optional
#     orientation="horizontal",
#     styles={
#         "container": {"padding": "5!important", "background-color": "#F0F2F6"},
#         "icon": {"color": "orange", "font-size": "25px"},
#         "nav-link": {
#             "font-size": "25px",
#             "text-align": "center",
#             "margin": "0px",
#             "--hover-color": "#eee",
#         },
#         "nav-link-selected": {"background-color": "green"},
#     },
# )


# Load your data
(df, df_customers_raw, df_orders_raw, df_transactions_raw, monthly_revenue, quarterly_revenue, customer_reten_monthly, 
            customer_reten_quarterly, churn_rate_monthly, churn_rate_quarterly, cltv, customer_churn_probability, 
            rfm_table, rfm_customer_segment, rfm_segment_final, product_affinity) = load_data()


# Sidebar filters
st.sidebar.header("Filters")
df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
start_date = st.sidebar.date_input("Start Date", df['Invoice Date'].min())
end_date = st.sidebar.date_input("End Date", df['Invoice Date'].max())

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

df_filtered = df[(df['Invoice Date'] >= start_date) & (df['Invoice Date'] <= end_date)]

st.session_state['df_filtered'] = df_filtered
st.session_state['start_date'] = start_date
st.session_state['end_date'] = end_date

st.title('Riffaut.club')
st.title("Customer Analytics Dashboard")
st.write("Welcome to the Customer Analytics Dashboard. Use the sidebar to navigate between different sections.")

st.header('Key Metrics')

df_filtered = st.session_state['df_filtered']

# Calculate metrics
customer_count = df_filtered['Customer ID'].nunique()  # Assuming 'CustomerID' is in df
target_customers = 120  # Example target
total_revenue = df_filtered['Amount'].sum()  # Assuming 'Revenue' is a column in df
target_revenue = 5000  # Example target revenue
average_order_value = total_revenue / customer_count
target_aov = 60  # Example target AOV


# Calculate progress
progress = customer_count / target_customers

# Header and progress bar
st.header('Customer Target Progress')

# Determine the delta and set the color conditionally
delta_value = target_customers - customer_count

if delta_value > 0:
    delta_text = f"{delta_value} behind target"
    delta_color = "orange"
else:
    delta_text = f"{abs(delta_value)} ahead of target"
    delta_color = "green"

# Display custom progress bar
progress_color = "red" if progress < 0.5 else "orange" if progress < 0.75 else "green"
st.markdown(f"""
    <div style="background-color: lightgrey; border-radius: 5px; padding: 3px;">
        <div style="width: {progress * 100}%; background-color: {progress_color}; height: 24px; border-radius: 5px;">
        </div>
    </div>
    """, unsafe_allow_html=True)

# Display the main metric with larger size
st.markdown(f"""
    <div style="text-align: center; font-size: 48px; font-weight: bold;">
        Total Customers: {customer_count:,} / {target_customers:,}
    </div>
    """, unsafe_allow_html=True)

# Display the delta text
st.markdown(
    f'<p style="color:{delta_color}; font-size: 20px; text-align: center;">{delta_text}</p>',
    unsafe_allow_html=True
)

# Create columns for the gauge charts
cola, colb = st.columns([1, 1])

# Gauge chart for total revenue
with cola:
    fig_revenue = go.Figure(go.Indicator(
        mode="gauge+number",
        value=total_revenue,
        title={'text': "Total Revenue (£)"},
        gauge={
            'axis': {'range': [None, target_revenue]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, target_revenue * 0.75], 'color': "lightgray"},
                {'range': [target_revenue * 0.75, target_revenue], 'color': "yellow"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': target_revenue
            }
        }
    ))
    st.plotly_chart(fig_revenue)

# Gauge chart for average order value
with colb:
    fig_aov = go.Figure(go.Indicator(
        mode="gauge+number",
        value=average_order_value,
        title={'text': "Average Order Value (£)"},
        gauge={
            'axis': {'range': [None, target_aov]},
            'bar': {'color': "blue"},
            'steps': [
                {'range': [0, target_aov * 0.75], 'color': "lightgray"},
                {'range': [target_aov * 0.75, target_aov], 'color': "yellow"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': target_aov
            }
        }
    ))
    st.plotly_chart(fig_aov)