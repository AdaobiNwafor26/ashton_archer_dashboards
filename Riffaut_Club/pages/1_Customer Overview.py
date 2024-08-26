import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
import numpy as np
from home import *


import streamlit as st
import pandas as pd
import plotly.express as px
import openai

# --- 1. OpenAI API Key Setup ---
client = OpenAI(api_key = 'sk-DKMHfpGtWv0HLU1Qze8dx6LcMXe_pbrha8vj1Atc1fT3BlbkFJqLsC-xrbiCWA5VpOhdlBgZdO7c3UCWMnZeZLjziHYA')

# --- 2. Functions for Modularization ---


# --- 2.1.1 Churn  ---


# Calculate churn insights (average, max, min, etc.)
def calculate_churn_insights(churn_data, period_label):
    average_churn_rate = churn_data['Churned'].mean()
    max_churn_rate = churn_data['Churned'].max()
    min_churn_rate = churn_data['Churned'].min()
    max_churn_period = churn_data[churn_data['Churned'] == max_churn_rate][period_label].dt.strftime('%Y-%m').values[0]
    min_churn_period = churn_data[churn_data['Churned'] == min_churn_rate][period_label].dt.strftime('%Y-%m').values[0]
    return average_churn_rate, max_churn_rate, min_churn_rate, max_churn_period, min_churn_period

# Display the calculated churn KPIs
def display_kpi_churn_metrics(average, max_rate, min_rate, max_period, min_period):
    st.header('Customer Churn Insights')
    col1, col2, col3 = st.columns(3)
    col1.metric("📉 Average Churn Rate", f"{average:.2f}%")
    col2.metric("🔺 Highest Churn Rate", f"{max_rate:.2f}%", f"in {max_period}")
    col3.metric("🔻 Lowest Churn Rate", f"{min_rate:.2f}%", f"in {min_period}")

# Create a line chart for churn data
def create_churn_line_chart(churn_data, period_label, title, marker_shape='circle', line_color='lime'):
    fig = px.line(churn_data, x=period_label, y='Churned', title=title,
                  markers=True, line_shape='linear', labels={'Churned': 'Churn Rate (%)', period_label: period_label},
                  color_discrete_sequence=['red'])

    fig.update_traces(marker=dict(size=10, symbol=marker_shape, color='red'), line=dict(color=line_color, width=3))
    fig.update_layout(title_text=title, title_x=0.5)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinewidth=1, zerolinecolor='black')

    st.plotly_chart(fig, use_container_width=True)


# --- 2.1.2 Retention  ---


# Calculate Retention Insights
def calculate_reten_insights(reten_data, period_label):
    if not pd.api.types.is_datetime64_any_dtype(reten_data[period_label]):
        reten_data[period_label] = pd.to_datetime(reten_data[period_label])

    highest_retention = reten_data['Retention Rate'].max()
    lowest_retention = reten_data['Retention Rate'].min()
    average_retention = reten_data['Retention Rate'].mean()

    max_reten_period = reten_data[reten_data['Retention Rate'] == highest_retention][period_label].dt.strftime('%Y-%m').values[0]
    min_reten_period = reten_data[reten_data['Retention Rate'] == lowest_retention][period_label].dt.strftime('%Y-%m').values[0]
    return highest_retention, lowest_retention, average_retention, max_reten_period, min_reten_period


# Display calculated retention KPIs
def display_kpi_reten_metrics(average_retention, highest_retention, lowest_retention, min_reten_period, max_reten_period):
    st.header('Customer Retention Insights')
    cola, colb, colc = st.columns(3)
    cola.metric("⤴︎ Highest Retention Rate", f"{highest_retention:.2f}%", f'in {max_reten_period}')
    colc.metric("⤵︎ Lowest Retention Rate", f"{lowest_retention:.2f}%", f'in {min_reten_period}')
    colb.metric("📈Average Retention Rate", f"{average_retention:.2f}%")

def create_reten_bar_chart(reten_data, period_label, title, bar_color='orange'):
    fig = px.bar(reten_data, x=period_label, y='Retention Rate', title=f'{title}',
                labels={'Retention Rate': 'Retention Rate (%)', period_label:period_label},
                color_discrete_sequence=[bar_color])
    
   # Customize the appearance of the bars
    fig.update_traces(marker_line_color='black', marker_line_width=1.5)  # Set border color and width


    fig.update_layout(title_text=title, title_x=0.5)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinewidth=1, zerolinecolor='black')

    st.plotly_chart(fig, use_container_width=True)


# --- 2.1.3 CLTV  ---


# Calculate CLTV Insights
def calculate_cltv_insights(cltv_data, period_label):
    if not pd.api.types.is_datetime64_any_dtype(cltv_data[period_label]):
        cltv_data[period_label] = pd.to_datetime(cltv_data[period_label])

    highest_cltv = cltv_data['CLTV'].max() * 100
    lowest_cltv = cltv_data['CLTV'].min() * 100
    average_cltv = cltv_data['CLTV'].mean() * 100

    max_cltv_period = cltv_data.loc[cltv_data['CLTV'].idxmax(), 'Last Purchase Date'].strftime('%Y-%m')
    min_cltv_period = cltv_data.loc[cltv_data['CLTV'].idxmin(), 'Last Purchase Date'].strftime('%Y-%m')

    # max_cltv_period = cltv_data[cltv_data['CLTV'] == highest_cltv][period_label].dt.strftime('%Y-%m').values[0]
    # min_cltv_period = cltv_data[cltv_data['CLTV'] == lowest_cltv][period_label].dt.strftime('%Y-%m').values[0]
    
    return highest_cltv, lowest_cltv, average_cltv, max_cltv_period, min_cltv_period


# Display calculated cltv KPIs
def display_kpi_cltv_metrics(average_cltv, highest_cltv, lowest_cltv, min_cltv_period, max_cltv_period):
    st.header('CLTV (Customer Lifetime Value) Insights')
    cold, cole, colf = st.columns(3)
    colf.metric("⤵︎ Lowest CLTV", f"{highest_cltv:.2f}%", f'in {max_cltv_period}')
    cold.metric("⤴︎ Highest CLTV", f"{lowest_cltv:.2f}%", f'in {min_cltv_period}')
    cole.metric("📈Average CLTV", f"{average_cltv:.2f}%")

def create_cltv_scatter_chart(cltv_data, period_label, title, bar_color='violet'):
    fig = px.scatter(cltv_data, x=period_label, y='CLTV', size='Total Revenue', title=f'{title}',
                labels={'CLTV': 'CLTV', period_label:period_label},
                color='Total Revenue',
                size_max=45,
                color_continuous_scale=['#FFD700', '#FF6347', '#FF4500'])
    
   # Customize the appearance of the bars
    fig.update_traces(marker_line_color='black', marker_line_width=1.5)  # Set border color and width


    fig.update_layout(title_text=title, title_x=0.5, legend_title_text='Total Revenue (£)')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='lightgrey', zeroline=True)

    st.plotly_chart(fig, use_container_width=True)

def create_cltv_segment_bar_chart(segment_counts, title):
    fig = px.bar(segment_counts, x='CLTV Segment', y='Number of Customers', title=title, 
                labels={'Number of Customers': 'Number of Customers'}, color='CLTV Segment', 
                category_orders={'CLTV Segment': ['Low CLTV', 'Medium CLTV', 'High CLTV']})

    st.plotly_chart(fig, use_container_width=False)

def create_cltv_trend_line_chart(cltv_data, period_label, title):
    fig = px.line(cltv_data, x=period_label, y='CLTV', title=title,
                  labels={'CLTV': 'Customer Lifetime Value', period_label: period_label},
                  color_discrete_sequence=['#FFD700'])

    fig.update_layout(title_text=title, title_x=0.5)
    st.plotly_chart(fig, use_container_width=False)   



# --- 2.1.4 Customer Churn Probabiity  ---


def calculate_churn_probability_metrics(cust_churn_data, churn_column='churn_proba', cltv_column='CLTV', threshold=0.5):
    average_churn_probability = cust_churn_data[churn_column].mean()
    num_customers_at_risk = cust_churn_data[churn_column].apply(lambda x: x > threshold).sum()
    cltv_at_risk = cust_churn_data.loc[cust_churn_data[churn_column] > threshold, cltv_column].sum()
    total_cltv = cust_churn_data[cltv_column].sum()
    percentage_cltv_at_risk = (cltv_at_risk / total_cltv) * 100 if total_cltv > 0 else 0

    return average_churn_probability, num_customers_at_risk, cltv_at_risk, percentage_cltv_at_risk



def display_kpi_churn_probability(average_churn_probability, num_customers_at_risk, cltv_at_risk, percentage_cltv_at_risk):
    st.header('Customer Churn Probability Insights')
    colg, colj, colh, coli = st.columns(4)
    colj.metric("📊 Avg Churn Probability", f"{average_churn_probability:.2f}")
    colg.metric("🚩 Customers at Risk", f"{num_customers_at_risk}")
    colh.metric("💰 CLTV at Risk", f"£{cltv_at_risk:.2f}")
    coli.metric("📉 % of CLTV at Risk", f"{percentage_cltv_at_risk:.2f}%")


def create_customer_segment_distribution_bar_chart(churn_segment_metrics, title):
    fig = px.bar(churn_segment_metrics, x='Cluster_Kmeans_Label', y='Number of Customers',
                title='Customer Distribution Across Segments',
                labels={'Number of Customers': 'Number of Customers'},
                color='Cluster_Kmeans_Label')

    st.plotly_chart(fig, use_container_width=False)


def create_customer_churn_risk_distribution_bar_chart_chart(churn_prob_segment_metrics, title):
    fig = px.bar(churn_prob_segment_metrics, x='churn_risk', y='Number of Customers',
                title='Customer Distribution Across Churn Risk Segments',
                labels={'Number of Customers': 'Number of Customers'},
                color='churn_risk')

    st.plotly_chart(fig, use_container_width=False)



def create_churn_probability_distribution(cust_churn_data, churn_column='churn_proba'):
    fig = px.histogram(cust_churn_data, x=churn_column, nbins=50, title='Churn Probability Distribution',
                    labels={churn_column: 'Churn Probability'},
                    color_discrete_sequence=['#FF6347'])

    fig.update_layout(title_text='Distribution of Customer Churn Probability', title_x=0.5)
    st.plotly_chart(fig, use_container_width=False)


def create_churn_probability_vs_cltv_scatter(cust_churn_data, churn_column='churn_proba', cltv_column='CLTV'):
    fig = px.scatter(cust_churn_data, x=churn_column, y=cltv_column, 
                    title='Churn Probability vs CLTV',
                    labels={churn_column: 'Churn Probability', cltv_column: 'CLTV'},
                    color=cltv_column, 
                    size=cltv_column, 
                    hover_data=['Customer ID'],
                    color_continuous_scale=px.colors.sequential.Viridis
)

    fig.update_layout(title_text='Churn Probability vs Customer Lifetime Value', title_x=0.5)

    st.plotly_chart(fig, use_container_width=False)

def create_churn_probability_heatmap(cust_churn_data, churn_column='churn_proba', cltv_column='CLTV'):
    fig = px.density_heatmap(cust_churn_data, x=churn_column, y=cltv_column, 
                            title='Churn Probability vs CLTV Heatmap',
                            labels={churn_column: 'Churn Probability', cltv_column: 'CLTV'},
                            color_continuous_scale='Plasma')

    fig.update_layout(title_text='Heatmap of Churn Probability and CLTV', title_x=0.5)
    st.plotly_chart(fig, use_container_width=False)



# --- 2.2 ChatGPT Generation  ---


# Generate interpretation using OpenAI GPT
def generate_interpretation(data, period):
    prompt = f"""
    Here is the {period} churn rate data: {data}.
    Please provide an interpretation that highlights any significant trends, such as high or low churn rates, and suggests potential actions or insights that could be drawn from this data.
    Provide any suggestions you would give the business owner in order to solve this problem.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Use "gpt-4" if you have access to GPT-4
        messages=[
            {"role": "system", "content": "You are an expert data scientist with a client in your consultancy."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()


# --- 3. Session State Management ---


st.session_state['churn_rate_monthly'] = churn_rate_monthly
st.session_state['churn_rate_quarterly'] = churn_rate_quarterly
st.session_state['customer_reten_monthly'] = customer_reten_monthly
st.session_state['customer_reten_quarterly'] = customer_reten_quarterly
st.session_state['cltv'] = cltv
st.session_state['customer_churn_probability'] = customer_churn_probability
st.session_state['rfm_segment_final'] = rfm_segment_final

# --- 4. Main App Logic ---


# Handle the data selection and display

tab1, tab2, tab3, tab4 = st.tabs(
    [
    'Churn',
    'Retention',
    # 'Monthly Churn',
    #  'Quarterly Churn',
    #  'Monthly Retention', 
    #  'Quarterly Retention',
     'Customer Churn Probability',
     'CLTV']
)


with tab1:
    st.subheader('Churn Analytics')

    stab11, stab12 = st.tabs(
        ['Monthly Churn',
         'Quarterly Churn']
    )

    with stab11:

        data = pd.DataFrame(st.session_state['churn_rate_monthly'])
        data['Month'] = pd.to_datetime(data['Month'])
        period_label = "Month"
        title = "Monthly Customer Churn Rate"


        # Calculate and display churn KPIs
        average_churn_rate, max_churn_rate, min_churn_rate, max_churn_period, min_churn_period = calculate_churn_insights(data, period_label)
        display_kpi_churn_metrics(average_churn_rate, max_churn_rate, min_churn_rate, max_churn_period, min_churn_period)
        
        # Display the churn rate line chart
        create_churn_line_chart(data, period_label, title)

        # Button to trigger insights
        if st.button("Generate Insights", key='insights_churn_monthly'):
            data_str = data.to_string(index=False)
            insights = generate_interpretation(data_str, period_label)
            st.markdown("### Insights")
            st.markdown(insights)

    with stab12:

        data = pd.DataFrame(st.session_state['churn_rate_quarterly'])
        data['Quarter'] = pd.to_datetime(data['Quarter'])
        period_label = "Quarter"
        title = "Quarterly Customer Churn Rate"

        # Calculate and display churn KPIs
        average_churn_rate, max_churn_rate, min_churn_rate, max_churn_period, min_churn_period = calculate_churn_insights(data, period_label)
        display_kpi_churn_metrics(average_churn_rate, max_churn_rate, min_churn_rate, max_churn_period, min_churn_period)
        
        # Display the churn rate line chart
        create_churn_line_chart(data, period_label, title)

        # Button to trigger insights
        if st.button("Generate Insights", key='insights_churn_quarterly'):
            data_str = data.to_string(index=False)
            insights = generate_interpretation(data_str, period_label)
            st.markdown("### Insights")
            st.markdown(insights)

with tab2:
    st.subheader('Retention Analytics')

    stab21, stab22 = st.tabs([
        'Monthly Retention Rate',
        'Quarterly Retention Rate'
    ])

    with stab21:
        data = pd.DataFrame(st.session_state['customer_reten_monthly'])
        data['Month'] = pd.to_datetime(data['Month'])
        period_label = "Month"
        title = "Monthly Customer Retention Rate"

        # Calculate and display retention KPIs
        highest_retention, lowest_retention, average_retention, min_reten_period, max_reten_period = calculate_reten_insights(data, period_label)
        display_kpi_reten_metrics(average_retention, highest_retention, lowest_retention, min_reten_period, max_reten_period)
        
        # Display the retention rate bar chart
        create_reten_bar_chart(data, period_label, title)

        # Button to trigger insights
        if st.button("Generate Insights", key='insights_reten_monthly'):
            data_str = data.to_string(index=False)
            insights = generate_interpretation(data_str, period_label)
            st.markdown("### Insights")
            st.markdown(insights)   

    with stab22:
        data = pd.DataFrame(st.session_state['customer_reten_quarterly'])
        data['Quarter'] = pd.to_datetime(data['Quarter'])
        data['Quarter Label'] = data['Quarter'].dt.to_period('Q').astype(str)
        period_label = "Quarter Label"
        title = "Quarterly Customer Retention Rate"

        # Calculate and display retention KPIs
        highest_retention, lowest_retention, average_retention, min_reten_period, max_reten_period = calculate_reten_insights(data, period_label)
        display_kpi_reten_metrics(average_retention, highest_retention, lowest_retention, min_reten_period, max_reten_period)
        
        # Display the retention rate bar chart
        create_reten_bar_chart(data, period_label, title)

        # Button to trigger insights
        if st.button("Generate Insights", key='insights_reten_quarterly'):
            data_str = data.to_string(index=False)
            insights = generate_interpretation(data_str, period_label)
            st.markdown("### Insights")
            st.markdown(insights)




with tab3:

    cust_churn_data = pd.DataFrame(st.session_state['rfm_segment_final'])
    cltv_data = pd.DataFrame(st.session_state['cltv'])

    # merging the churn prob and cltv data
    data = pd.merge(cltv_data, cust_churn_data, on='Customer ID')
    period_label = 'Customer ID'
    title = 'Customer Churn Probability and CLTV'

    churn_segment_metrics = data.groupby('Cluster_Kmeans_Label').agg({
        # 'Monetary':'mean',
        'churn_proba':'mean',
        'Customer ID':'count'
    }).reset_index()

    churn_segment_metrics.rename(columns={
        # 'Monetary':'Average Monetary',
        'churn_proba': 'Average Churn Probability',
        'Customer ID': 'Number of Customers'
    }, inplace=True)

    churn_prob_segment_metrics = data.groupby('churn_risk').agg({
        # 'Monetary':'mean',
        'churn_proba':'mean',
        'Customer ID':'count'
    }).reset_index()

    churn_prob_segment_metrics.rename(columns={
        # 'Monetary':'Average Monetary',
        'churn_proba':'Average Churn Probability',
        'Customer ID':'Number of Customers'
    }, inplace=True)

    average_churn_probability, num_customers_at_risk, cltv_at_risk, percentage_cltv_at_risk = calculate_churn_probability_metrics(data)
    display_kpi_churn_probability(average_churn_probability, num_customers_at_risk, cltv_at_risk, percentage_cltv_at_risk)

    colk, coll = st.columns([1,1])

    with colk:
        create_customer_segment_distribution_bar_chart(churn_segment_metrics, title)
    with coll:
        create_customer_churn_risk_distribution_bar_chart_chart(churn_prob_segment_metrics, title)

    create_churn_probability_heatmap(data)

    colm, coln = st.columns(2)

    with colm:
        create_churn_probability_distribution(data)
    
    with coln:
        create_churn_probability_vs_cltv_scatter(data)

        # Button to trigger insights
    if st.button("Generate Insights", key='insights_churn_prob'):
        data_str = data.to_string(index=False)
        insights = generate_interpretation(data_str, period_label)
        st.markdown("### Insights")
        st.markdown(insights)



with tab4:

    data = pd.DataFrame(st.session_state['cltv'])
    data['Last Purchase Date'] = pd.to_datetime(data['Last Purchase Date'], format='%Y-%m-%d')
    data['Customer Lifespan'] = (data['Last Purchase Date'] - data['First Purchase Date']).dt.days
    bins = [0, 0.2, 0.5, 1.0]  # Example thresholds for segmentation
    labels = ['Low CLTV', 'Medium CLTV', 'High CLTV']
    data['CLTV Segment'] = pd.cut(data['CLTV'], bins=bins, labels=labels)
    period_label = 'Last Purchase Date'
    title = 'Last Purchase Date vs CLTV'

    highest_cltv, lowest_cltv, average_cltv, min_cltv_period, max_cltv_period = calculate_cltv_insights(data, period_label)
    display_kpi_cltv_metrics(highest_cltv, lowest_cltv, average_cltv, min_cltv_period, max_cltv_period)

    create_cltv_scatter_chart(data, period_label, title)

    segment_counts = data.groupby('CLTV Segment').size().reset_index(name='Number of Customers')

    create_cltv_segment_bar_chart(segment_counts, 'Customer Segmentation by CLTV')
    # create_cltv_trend_line_chart(data, period_label, 'CLTV Over Time')

    # Button to trigger insights
    if st.button("Generate Insights", key='insights_cltv'):
        data_str = data.to_string(index=False)
        insights = generate_interpretation(data_str, period_label)
        st.markdown("### Insights")
        st.markdown(insights)

