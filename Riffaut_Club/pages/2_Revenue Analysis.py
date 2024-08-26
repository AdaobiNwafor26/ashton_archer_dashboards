import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from home import *
from openai import OpenAI
import numpy as np


import streamlit as st
import pandas as pd
import plotly.express as px
import openai

# --------- 1. OpenAI API Key Setup  ------------


# --------- 2. Functions for Modularization  ------------

# --------- 2.1.1 Revenue  ------------

def calculate_revenue_insights(revenue_data, period_label):

    revenue_data['Invoice Date'] = pd.to_datetime(revenue_data['Invoice Date'])
    print(revenue_data['Invoice Date'].max())

    if not pd.api.types.is_datetime64_any_dtype(revenue_data[period_label]):
        revenue_data[period_label] = pd.to_datetime(revenue_data[period_label])

    average_revenue = revenue_data['Amount'].mean()
    max_revenue = revenue_data['Amount'].max()
    min_revenue = revenue_data['Amount'].min()
    total_revenue = revenue_data['Amount'].sum()
    
    # Assuming the Invoice Date is a datetime object
    max_revenue_period = revenue_data[revenue_data['Amount'] == max_revenue][period_label].dt.strftime('%Y-%m').values[0]
    min_revenue_period = revenue_data[revenue_data['Amount'] == min_revenue][period_label].dt.strftime('%Y-%m').values[0]
    
    return average_revenue, max_revenue, min_revenue, total_revenue, max_revenue_period, min_revenue_period

def display_kpi_revenue_metrics(average_revenue, max_revenue, min_revenue, total_revenue, max_period, min_period):
    st.header('Revenue Insights')
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("🔺 Highest Revenue", f"${max_revenue:,.2f}", f"in {max_period}")
    col3.metric("🔻 Lowest Revenue", f"${min_revenue:,.2f}", f"in {min_period}")
    st.metric("📉 Average Revenue", f"${average_revenue:,.2f}")

def create_revenue_line_chart(revenue_data, period_label, title, marker_shape='circle', line_color='green'):
    fig = px.line(revenue_data, x=period_label, y='Amount', title=title,
                  markers=True, line_shape='linear', labels={'Amount': 'Revenue ($)', period_label: period_label},
                  color_discrete_sequence=[line_color])

    fig.update_traces(marker=dict(size=10, symbol=marker_shape, color=line_color), line=dict(color=line_color, width=3))
    fig.update_layout(title_text=title, title_x=0.5)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinewidth=1, zerolinecolor='black')

    st.plotly_chart(fig, use_container_width=True)



# --------- 2.1.2 Revenue - Growth Rate ------------


def get_current_and_previous_period_revenue(revenue_data, period_label):
    # Ensure the period_label column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(revenue_data[period_label]):
        revenue_data[period_label] = pd.to_datetime(revenue_data[period_label])

    # Get the most recent period
    current_period = revenue_data[period_label].max()
    current_quarter = current_period.quarter

    # Filter for the current period
    current_period_data = revenue_data[(revenue_data[period_label].dt.quarter == current_quarter)]

    # Determine the previous period
    if current_quarter == 1:
        previous_quarter = 4
    else:
        previous_quarter = current_quarter - 1

    # Filter for the previous period
    previous_period_data = revenue_data[(revenue_data[period_label].dt.quarter == previous_quarter)]

    # Calculate total revenue for each period
    current_revenue = current_period_data['Amount'].sum()
    previous_revenue = previous_period_data['Amount'].sum()

    return current_revenue, previous_revenue


def calculate_revenue_growth_rate(current_revenue, previous_revenue):
    return (current_revenue - previous_revenue) / previous_revenue * 100


def display_revenue_growth_bullet_graph(current_revenue, previous_revenue, goal):
    growth_rate = calculate_revenue_growth_rate(current_revenue, previous_revenue)
    fig = go.Figure(go.Indicator(
        mode = "number+gauge+delta",
        value = growth_rate,
        delta = {'reference': goal},
        gauge = {
            'shape': "bullet",
            'axis': {'range': [0, 2 * goal]},
            'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': goal},
            'bgcolor': "white",
            'steps': [{'range': [0, goal], 'color': "lightgray"}]
        },
        title = {'text': "Revenue Growth Rate"}
    ))
    st.plotly_chart(fig)


# --------- 2.1.3 Product Affinity Analysis --------- 

def calculate_product_affinity_insights(affinity_data):
    most_purchased_product = affinity_data.groupby('Product Name')['Product Purchases Made'].sum().idxmax()
    highest_affinity_score = affinity_data['Product Affinity'].max()
    average_affinity_score = affinity_data['Product Affinity'].mean()
    total_purchases = affinity_data['Product Purchases Made'].sum()
    
    return most_purchased_product, highest_affinity_score, average_affinity_score, total_purchases

def display_kpi_product_affinity(most_purchased_product, highest_affinity_score, average_affinity_score, total_purchases):
    st.header('Product Affinity Insights')
    col5, col6, col7 = st.columns(3)
    st.metric("📈 Most Purchased Product", most_purchased_product)
    col5.metric("⭐ Highest Affinity Score", f"{highest_affinity_score:.2f}")
    col6.metric("📉 Average Affinity Score", f"{average_affinity_score:.2f}")
    col7.metric("🛒 Total Purchases", total_purchases)

def create_product_affinity_bar_chart(affinity_data):
    product_purchase_data = affinity_data.groupby('Product Name')['Product Purchases Made'].sum().reset_index()
    
    fig = px.bar(product_purchase_data, x='Product Name', y='Product Purchases Made', 
                 title='Total Purchases by Product', 
                 labels={'Product Purchases Made': 'Total Purchases'})
    
    st.plotly_chart(fig, use_container_width=True)


def create_co_occurrence_matrix(order_data):
    from mlxtend.frequent_patterns import association_rules, apriori
    from mlxtend.preprocessing import TransactionEncoder
    
    # Assuming each order is a list of products
    basket_sets = order_data.groupby('Order ID')['Product Name'].apply(list).tolist()
    
    te = TransactionEncoder()
    te_ary = te.fit_transform(basket_sets)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Use Apriori algorithm to find frequent itemsets
    frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    
    return rules

def create_product_association_heatmap(rules):
    pivot = rules.pivot(index='antecedents', columns='consequents', values='lift')
    fig = px.imshow(pivot, color_continuous_scale='Viridis', title="Product Association Heatmap")
    st.plotly_chart(fig, use_container_width=True)





# --------- 3. Session State Managenent  ------------

st.session_state['monthly_revenue_trends'] = monthly_revenue
st.session_state['quarterly_revenue_trends'] = quarterly_revenue
st.session_state['product_affinity'] = product_affinity
st.session_state['orders'] = df_orders_raw


# --------- 4. Main App Logic  ------------

tab1, tab2= st.tabs([
    'Revenue Analysis',
    'Product Affinity Analysis',
    # 'Advanced Analysis'
])

with tab1:
    st.subheader('Revenue Analytics')

    stab11, stab12 = st.tabs(
        ['Monthly Revenue',
        'Quarterly Revenue'
    ])

    with stab11:
        st.write('This is the Overview for Monthly Revenue')
        avg_rev, max_rev, min_rev, total_rev, max_period, min_period = calculate_revenue_insights(monthly_revenue, 'Invoice Date')
        display_kpi_revenue_metrics(avg_rev, max_rev, min_rev, total_rev, max_period, min_period)
        
        create_revenue_line_chart(monthly_revenue, 'Invoice Date', "Monthly Revenue Trend")
        
        current_revenue, previous_revenue = get_current_and_previous_period_revenue(monthly_revenue, 'Invoice Date')
        # display_revenue_growth_bullet_graph(current_revenue, previous_revenue, goal=15)



    with stab12:
        
        st.write('This is the Overview for Quarterly Revenue')

        avg_rev, max_rev, min_rev, total_rev, max_period, min_period = calculate_revenue_insights(quarterly_revenue, 'Invoice Date')
        display_kpi_revenue_metrics(avg_rev, max_rev, min_rev, total_rev, max_period, min_period)
        
        create_revenue_line_chart(quarterly_revenue, 'Invoice Date', "Quarterly Revenue Trend")

with tab2:
    st.header('Product Affinity Analytics')

    stab21, stab22, stab23 = st.tabs([
        'Affinity Overview',
        'Cross-Selling Opportunities',
        'Customer Segmentation'
    ])

    with stab21:
        st.write('This is the Overview for Product Affinity')
        most_purchased_product, highest_affinity_score, average_affinity_score, total_purchases = calculate_product_affinity_insights(product_affinity)
        display_kpi_product_affinity(most_purchased_product, highest_affinity_score, average_affinity_score, total_purchases)
        create_product_affinity_bar_chart(product_affinity)

    with stab22:
        # rules = create_co_occurrence_matrix(df_orders_raw)
        # create_product_association_heatmap(rules)
        pass

    






# with tab3:
#     st.header('Advanced Revenue Analysis')

#     # current_revenue, previous_revenue = get_current_and_previous_period_revenue()






