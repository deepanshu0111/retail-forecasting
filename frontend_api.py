"""
==============================================
RETAIL DEMAND FORECASTING - FRONTEND
==============================================
Streamlit Dashboard with Backend API Integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

# Configuration
API_BASE_URL = "http://localhost:5000"

# Page setup
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

# Custom styling
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    h1 {color: #1f77b4;}
    </style>
    """, unsafe_allow_html=True)

# Helper functions
def check_api():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_stats():
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        return response.json() if response.status_code == 200 else None
    except:
        return None

def forecast(store_id, item_id, start_date, days, price, promotion, holiday):
    try:
        payload = {
            'store_id': store_id,
            'item_id': item_id,
            'start_date': start_date,
            'days': days,
            'price': price,
            'promotion': promotion,
            'holiday': holiday
        }
        response = requests.post(f"{API_BASE_URL}/forecast", json=payload)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Main app
st.title("📊 Retail Demand Forecasting Dashboard")
st.markdown("### AI-Powered Demand Prediction")
st.markdown("---")

# Check API
if not check_api():
    st.error("⚠ *Backend API is not running!*")
    st.info("""
    Please start the backend first:
    
    python backend_api.py
    
    """)
    st.stop()

st.success("✅ Connected to Backend API")

# Get stats
stats = get_stats()

if stats:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏪 Stores", stats['stores']['count'])
    with col2:
        st.metric("📦 Items", stats['items']['count'])
    with col3:
        st.metric("📊 Records", f"{stats['total_records']:,}")
    with col4:
        st.metric("💰 Avg Sales", f"{stats['sales']['mean']:.1f}")

st.markdown("---")

# Sidebar
st.sidebar.header("⚙ Configuration")

selected_store = st.sidebar.selectbox("Store ID", range(1, 11), index=0)
selected_item = st.sidebar.selectbox("Item ID", range(1, 51), index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Forecast Settings")

forecast_days = st.sidebar.slider("Days to Forecast", 7, 90, 30, 7)

st.sidebar.subheader("🎯 Parameters")

price_input = st.sidebar.number_input(
    "Price ($)",
    min_value=10.0,
    max_value=200.0,
    value=50.0,
    step=5.0
)

promotion_input = st.sidebar.checkbox("Apply Promotion", value=False)
holiday_input = st.sidebar.checkbox("Include Holidays", value=False)

# Main content
st.subheader("🔮 Generate Forecast")

col1, col2 = st.columns([3, 1])

with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now().date() + timedelta(days=1),
        min_value=datetime.now().date()
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)

if generate_btn:
    with st.spinner("Generating forecast..."):
        result = forecast(
            store_id=selected_store,
            item_id=selected_item,
            start_date=start_date.isoformat(),
            days=forecast_days,
            price=price_input,
            promotion=int(promotion_input),
            holiday=int(holiday_input)
        )
        
        if result:
            st.success("✅ Forecast generated!")
            
            # Summary metrics
            st.markdown("### 📊 Forecast Summary")
            summary = result['summary']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Sales", f"{summary['total_predicted_sales']:.0f} units")
            with col2:
                st.metric("Avg Daily", f"{summary['average_daily_sales']:.1f} units")
            with col3:
                st.metric("Peak Day", f"{summary['max_daily_sales']:.0f} units")
            with col4:
                st.metric("Min Day", f"{summary['min_daily_sales']:.0f} units")
            
            # Visualization
            st.markdown("### 📈 Forecast Chart")
            forecast_df = pd.DataFrame(result['forecast'])
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
            
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(forecast_df['date'], forecast_df['predicted_sales'],
                   marker='o', linewidth=2, markersize=6, color='#1f77b4',
                   label='Predicted Sales')
            
            # Trend line
            z = np.polyfit(range(len(forecast_df)), forecast_df['predicted_sales'], 1)
            p = np.poly1d(z)
            ax.plot(forecast_df['date'], p(range(len(forecast_df))),
                   '--', color='red', linewidth=2, alpha=0.7, label='Trend')
            
            ax.fill_between(forecast_df['date'],
                           forecast_df['predicted_sales'] * 0.9,
                           forecast_df['predicted_sales'] * 1.1,
                           alpha=0.2, color='#1f77b4', label='Confidence Band')
            
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Predicted Sales', fontsize=12, fontweight='bold')
            ax.set_title(f'Demand Forecast - Store {selected_store}, Item {selected_item}',
                       fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Data table
            st.markdown("### 📋 Detailed Forecast")
            display_df = forecast_df.copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df['predicted_sales'] = display_df['predicted_sales'].round(1)
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Download
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"forecast_store{selected_store}_item{selected_item}.csv",
                "text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>📊 Retail Demand Forecasting System</strong></p>
    <p>Backend: Flask API | Frontend: Streamlit | ML: Random Forest</p>
</div>
""", unsafe_allow_html=True)