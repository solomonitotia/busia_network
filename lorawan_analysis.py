import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="LoRaWAN Network Analyzer Pro",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .status-good { border-left: 5px solid #2ecc71; background-color: #d5f4e6; }
    .status-warning { border-left: 5px solid #f39c12; background-color: #fef5e7; }
    .status-error { border-left: 5px solid #e74c3c; background-color: #fdf2f2; }
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Professional header
st.markdown('<h1 class="main-header">📡 LoRaWAN Network Analyzer Pro</h1>', unsafe_allow_html=True)

# Enhanced sidebar
st.sidebar.markdown("### 📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload LoRaWAN Data",
    type=['xlsx', 'xls', 'csv'],
    help="Upload your LoRaWAN uplinks file"
)

# Function to create enhanced metrics
def create_metric_card(title, value, delta=None, status="good"):
    status_class = f"status-{status}"
    delta_html = ""
    if delta:
        delta_color = "#2ecc71" if delta >= 0 else "#e74c3c"
        delta_html = f'<p style="color: {delta_color}; margin: 0; font-size: 0.8rem;">Δ {delta}</p>'
    
    return f"""
    <div class="info-box {status_class}">
        <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">{title}</p>
        <p style="margin: 0; font-size: 2rem; font-weight: bold;">{value}</p>
        {delta_html}
    </div>
    """

# Function to assess network health
def assess_network_health(df):
    if len(df) == 0:
        return "error", "No data available"
    
    strong_pct = (df['rssi'] > -70).mean() * 100
    weak_pct = (df['rssi'] < -90).mean() * 100
    
    gateway_counts = df['gateway_id'].value_counts()
    load_balance = gateway_counts.max() / gateway_counts.min() if len(gateway_counts) > 1 else 1
    
    if strong_pct > 80 and weak_pct < 10 and load_balance < 2:
        return "good", "Excellent network performance"
    elif strong_pct > 60 and weak_pct < 20 and load_balance < 3:
        return "warning", "Good performance with minor optimization opportunities"
    else:
        return "error", "Network performance issues detected"

if uploaded_file is not None:
    try:
        with st.spinner('Loading data...'):
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
        
        st.success(f"✅ Data loaded successfully! {len(df):,} records found")
        
        # Sidebar filters
        st.sidebar.header("🔍 Filters")
        
        available_devices = ['All Devices'] + list(df['device_id'].unique())
        selected_device = st.sidebar.selectbox("Select Device", available_devices)
        
        available_gateways = ['All Gateways'] + list(df['gateway_id'].unique())
        selected_gateway = st.sidebar.selectbox("Select Gateway", available_gateways)
        
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_device != 'All Devices':
            filtered_df = filtered_df[filtered_df['device_id'] == selected_device]
        
        if selected_gateway != 'All Gateways':
            filtered_df = filtered_df[filtered_df['gateway_id'] == selected_gateway]
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= start_date) & 
                (filtered_df['timestamp'].dt.date <= end_date)
            ]
        
        # Network health assessment
        health_status, health_message = assess_network_health(filtered_df)
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🌐 Gateway Analysis", 
            "📱 Device Analysis", 
            "📶 Signal Quality", 
            "⏱️ Time Analysis"
        ])
        
        # Tab 1: Overview
        with tab1:
            st.header("📊 Network Overview")
            
            # Network health indicator
            health_colors = {"good": "#2ecc71", "warning": "#f39c12", "error": "#e74c3c"}
            st.markdown(f"""
            <div style="background-color: {health_colors[health_status]}; color: white; padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 2rem;">
                <h3 style="margin: 0;">Network Health: {health_message}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_messages = len(filtered_df)
                st.markdown(create_metric_card("Total Messages", f"{total_messages:,}"), unsafe_allow_html=True)
            
            with col2:
                unique_devices = filtered_df['device_id'].nunique()
                st.markdown(create_metric_card("Unique Devices", unique_devices), unsafe_allow_html=True)
            
            with col3:
                unique_gateways = filtered_df['gateway_id'].nunique()
                st.markdown(create_metric_card("Unique Gateways", unique_gateways), unsafe_allow_html=True)
            
            with col4:
                avg_rssi = filtered_df['rssi'].mean()
                rssi_status = "good" if avg_rssi > -70 else "warning" if avg_rssi > -85 else "error"
                st.markdown(create_metric_card("Avg RSSI", f"{avg_rssi:.1f} dBm", status=rssi_status), unsafe_allow_html=True)
            
            # Overview charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Signal quality distribution
                strong_signals = (filtered_df['rssi'] > -70).sum()
                moderate_signals = ((filtered_df['rssi'] >= -90) & (filtered_df['rssi'] <= -70)).sum()
                weak_signals = (filtered_df['rssi'] < -90).sum()
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Strong (>-70 dBm)', 'Moderate (-70 to -90 dBm)', 'Weak (<-90 dBm)'],
                    values=[strong_signals, moderate_signals, weak_signals],
                    marker=dict(colors=['#2ecc71', '#f39c12', '#e74c3c'])
                )])
                fig_pie.update_layout(title="Signal Quality Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Device type distribution
                device_type_counts = filtered_df['device_type'].value_counts()
                fig_bar = px.bar(
                    x=device_type_counts.index,
                    y=device_type_counts.values,
                    title="Messages by Device Type"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Daily message trend
            daily_messages = filtered_df.groupby(filtered_df['timestamp'].dt.date).size().reset_index()
            daily_messages.columns = ['date', 'messages']
            
            fig_trend = px.line(
                daily_messages,
                x='date',
                y='messages',
                title="Daily Message Volume Trend"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        
        # Tab 2: Gateway Analysis
        with tab2:
            st.header("🌐 Gateway Performance Analysis")
            
            # Gateway performance metrics
            gateway_stats = filtered_df.groupby('gateway_id').agg({
                'rssi': ['mean', 'std', 'min', 'max', 'count'],
                'snr': ['mean', 'std'],
                'sf': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
            }).round(2)
            
            gateway_stats.columns = ['rssi_mean', 'rssi_std', 'rssi_min', 'rssi_max', 'message_count',
                                   'snr_mean', 'snr_std', 'common_sf']
            gateway_stats.reset_index(inplace=True)
            
            st.subheader("Gateway Performance Summary")
            st.dataframe(gateway_stats, use_container_width=True)
            
            # Gateway comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_box = px.box(
                    filtered_df,
                    x='gateway_id',
                    y='rssi',
                    title="RSSI Distribution by Gateway"
                )
                fig_box.update_xaxes(tickangle=45)
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                gateway_counts = filtered_df['gateway_id'].value_counts()
                fig_bar = px.bar(
                    x=gateway_counts.index,
                    y=gateway_counts.values,
                    title="Message Load by Gateway"
                )
                fig_bar.update_xaxes(tickangle=45)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Gateway performance heatmap
            gateway_hourly = filtered_df.groupby(['gateway_id', 'hour'])['rssi'].mean().reset_index()
            gateway_pivot = gateway_hourly.pivot(index='gateway_id', columns='hour', values='rssi')
            
            if not gateway_pivot.empty:
                fig_heatmap = px.imshow(
                    gateway_pivot,
                    title="Average RSSI by Gateway and Hour",
                    labels=dict(x="Hour of Day", y="Gateway ID", color="RSSI (dBm)"),
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Tab 3: Device Analysis
        with tab3:
            st.header("📱 Device Performance Analysis")
            
            device_stats = filtered_df.groupby(['device_id', 'device_type']).agg({
                'rssi': ['mean', 'std', 'min', 'max', 'count'],
                'snr': ['mean', 'std'],
                'gateway_id': 'nunique'
            }).round(2)
            
            device_stats.columns = ['rssi_mean', 'rssi_std', 'rssi_min', 'rssi_max', 'message_count',
                                  'snr_mean', 'snr_std', 'connected_gateways']
            device_stats.reset_index(inplace=True)
            
            st.subheader("Device Performance Summary")
            st.dataframe(device_stats, use_container_width=True)
            
            # Device-specific analysis
            if selected_device != 'All Devices':
                st.subheader(f"Detailed Analysis for Device: {selected_device}")
                
                device_data = filtered_df[filtered_df['device_id'] == selected_device]
                
                if len(device_data) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Messages Sent", len(device_data))
                    
                    with col2:
                        st.metric("Avg RSSI", f"{device_data['rssi'].mean():.1f} dBm")
                    
                    with col3:
                        st.metric("Avg SNR", f"{device_data['snr'].mean():.1f} dB")
                    
                    with col4:
                        st.metric("Connected Gateways", device_data['gateway_id'].nunique())
                    
                    # Device performance charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        daily_device_rssi = device_data.groupby(device_data['timestamp'].dt.date)['rssi'].mean().reset_index()
                        daily_device_rssi.columns = ['date', 'rssi']
                        
                        fig_device_trend = px.line(
                            daily_device_rssi,
                            x='date',
                            y='rssi',
                            title=f"Signal Quality Trend - {selected_device}"
                        )
                        st.plotly_chart(fig_device_trend, use_container_width=True)
                    
                    with col2:
                        device_gateway_counts = device_data['gateway_id'].value_counts()
                        fig_device_gw = px.pie(
                            values=device_gateway_counts.values,
                            names=device_gateway_counts.index,
                            title=f"Gateway Usage - {selected_device}"
                        )
                        st.plotly_chart(fig_device_gw, use_container_width=True)
            
            else:
                # All devices comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    device_performance = filtered_df.groupby('device_id')['rssi'].mean().reset_index()
                    fig_device_perf = px.bar(
                        device_performance,
                        x='device_id',
                        y='rssi',
                        title="Average RSSI by Device"
                    )
                    fig_device_perf.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_device_perf, use_container_width=True)
                
                with col2:
                    device_connectivity = filtered_df.groupby('device_id')['gateway_id'].nunique().reset_index()
                    device_connectivity.columns = ['device_id', 'gateway_count']
                    
                    fig_connectivity = px.bar(
                        device_connectivity,
                        x='device_id',
                        y='gateway_count',
                        title="Gateway Connectivity by Device"
                    )
                    fig_connectivity.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_connectivity, use_container_width=True)
        
        # Tab 4: Signal Quality Analysis
        with tab4:
            st.header("📶 Signal Quality Analysis")
            
            # Signal quality metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average RSSI", f"{filtered_df['rssi'].mean():.1f} dBm")
                st.metric("RSSI Range", f"{filtered_df['rssi'].min():.1f} to {filtered_df['rssi'].max():.1f} dBm")
            
            with col2:
                st.metric("Average SNR", f"{filtered_df['snr'].mean():.1f} dB")
                st.metric("SNR Range", f"{filtered_df['snr'].min():.1f} to {filtered_df['snr'].max():.1f} dB")
            
            with col3:
                quality_score = (filtered_df['rssi'] > -70).mean() * 100
                st.metric("Quality Score", f"{quality_score:.1f}%")
                weak_pct = (filtered_df['rssi'] < -90).mean() * 100
                st.metric("Weak Signals", f"{weak_pct:.1f}%")
            
            # Signal quality charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    filtered_df,
                    x='rssi',
                    nbins=50,
                    title="RSSI Distribution"
                )
                fig_hist.add_vline(x=-70, line_dash="dash", line_color="green", annotation_text="Good")
                fig_hist.add_vline(x=-90, line_dash="dash", line_color="red", annotation_text="Poor")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_scatter = px.scatter(
                    filtered_df,
                    x='rssi',
                    y='snr',
                    color='gateway_id',
                    title="RSSI vs SNR Analysis"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Spreading factor analysis
            if 'sf' in filtered_df.columns:
                sf_usage = filtered_df['sf'].value_counts().sort_index()
                fig_sf = px.bar(
                    x=sf_usage.index,
                    y=sf_usage.values,
                    title="Spreading Factor Usage"
                )
                st.plotly_chart(fig_sf, use_container_width=True)
        
        # Tab 5: Time Analysis
        with tab5:
            st.header("⏱️ Time-based Analysis")
            
            # Time-based charts
            col1, col2 = st.columns(2)
            
            with col1:
                hourly_dist = filtered_df['hour'].value_counts().sort_index()
                fig_hourly = px.bar(
                    x=hourly_dist.index,
                    y=hourly_dist.values,
                    title="Message Distribution by Hour"
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
            
            with col2:
                dow_dist = filtered_df['day_of_week'].value_counts()
                fig_dow = px.bar(
                    x=dow_dist.index,
                    y=dow_dist.values,
                    title="Message Distribution by Day of Week"
                )
                st.plotly_chart(fig_dow, use_container_width=True)
            
            # Signal quality over time
            if len(filtered_df) > 0:
                hourly_signal = filtered_df.groupby('hour')['rssi'].mean().reset_index()
                fig_signal_time = px.line(
                    hourly_signal,
                    x='hour',
                    y='rssi',
                    title="Average Signal Quality by Hour"
                )
                st.plotly_chart(fig_signal_time, use_container_width=True)
                
                # Daily signal quality by gateway
                if filtered_df['gateway_id'].nunique() > 1:
                    daily_gateway_signal = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'gateway_id'])['rssi'].mean().reset_index()
                    daily_gateway_signal.columns = ['date', 'gateway_id', 'rssi']
                    
                    fig_daily_gw = px.line(
                        daily_gateway_signal,
                        x='date',
                        y='rssi',
                        color='gateway_id',
                        title="Daily Signal Quality by Gateway"
                    )
                    st.plotly_chart(fig_daily_gw, use_container_width=True)
        
        # Download section
        st.sidebar.header("💾 Export Data")
        
        if st.sidebar.button("Download Filtered Data"):
            csv = filtered_df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"lorawan_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.info("Please make sure your Excel file has the correct format.")

else:
    st.info("👆 Please upload your LoRaWAN Excel file using the sidebar to start analysis")
    
    st.markdown("""
    ## 🚀 What this app does:
    
    ### 📊 **Overview Dashboard**
    - Key network metrics and KPIs
    - Signal quality distribution
    - Daily traffic trends
    
    ### 🌐 **Gateway Analysis**
    - Performance comparison between gateways
    - Load balancing analysis
    - Coverage patterns
    
    ### 📱 **Device Analysis**
    - Individual device performance
    - Device connectivity patterns
    - Signal quality trends per device
    
    ### 📶 **Signal Quality**
    - RSSI and SNR analysis
    - Quality score calculations
    - Spreading factor usage
    
    ### ⏱️ **Time Analysis**
    - Traffic patterns by hour/day
    - Signal quality trends over time
    - Peak usage identification
    """)