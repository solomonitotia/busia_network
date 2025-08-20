import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="LoRaWAN Network",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional color scheme
COLORS = {
    'primary': '#2E4057',      # Dark blue-gray
    'secondary': '#546A7B',    # Medium blue-gray  
    'accent': '#9EA3B0',       # Light blue-gray
    'success': '#28A745',      # Professional green
    'warning': '#FFC107',      # Professional amber
    'danger': '#DC3545',       # Professional red
    'light': '#F8F9FA',        # Light gray
    'white': '#FFFFFF',        # White
    'text': '#333333',         # Dark text
    'muted': '#6C757D'         # Muted text
}

# Professional CSS styling
st.markdown(f"""
<style>
    /* Main header styling */
    .main-header {{
        font-size: 3rem;
        color: {COLORS['primary']};
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    
    /* Professional metric cards */
    .metric-card {{
        background: {COLORS['white']};
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #E9ECEF;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s ease;
    }}
    
    .metric-card:hover {{
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: 600;
        margin: 0.5rem 0;
        color: {COLORS['primary']};
    }}
    
    .metric-label {{
        font-size: 0.875rem;
        color: {COLORS['muted']};
        margin: 0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .metric-delta {{
        font-size: 0.8rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }}
    
    /* Status indicators */
    .status-excellent {{ 
        border-left: 4px solid {COLORS['success']};
        background: linear-gradient(90deg, rgba(40, 167, 69, 0.05) 0%, {COLORS['white']} 100%);
    }}
    .status-good {{ 
        border-left: 4px solid {COLORS['warning']};
        background: linear-gradient(90deg, rgba(255, 193, 7, 0.05) 0%, {COLORS['white']} 100%);
    }}
    .status-poor {{ 
        border-left: 4px solid {COLORS['danger']};
        background: linear-gradient(90deg, rgba(220, 53, 69, 0.05) 0%, {COLORS['white']} 100%);
    }}
    
    /* Professional tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        background-color: {COLORS['light']};
        border-radius: 6px;
        padding: 0.25rem;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 48px;
        padding: 0 20px;
        background: {COLORS['white']};
        border-radius: 4px;
        border: 1px solid transparent;
        font-weight: 500;
        color: {COLORS['text']};
        transition: all 0.2s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: {COLORS['accent']};
        color: {COLORS['white']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {COLORS['primary']};
        color: {COLORS['white']};
        border-color: {COLORS['primary']};
    }}
    
    /* Health indicator */
    .health-indicator {{
        padding: 1rem 2rem;
        border-radius: 6px;
        text-align: center;
        margin-bottom: 2rem;
        color: {COLORS['white']};
        font-weight: 600;
        font-size: 1.1rem;
    }}
    
    /* Professional info boxes */
    .info-box {{
        background: {COLORS['white']};
        border-radius: 6px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #E9ECEF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    
    /* Gateway connection cards */
    .gateway-connection {{
        background: {COLORS['white']};
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid {COLORS['primary']};
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }}
    
    .gateway-connection:hover {{
        transform: translateX(2px);
    }}
    
    /* Distance indicators */
    .distance-close {{ 
        background: {COLORS['success']}; 
        color: white; 
        padding: 0.25rem 0.75rem; 
        border-radius: 12px; 
        font-size: 0.75rem; 
        font-weight: 500;
    }}
    .distance-medium {{ 
        background: {COLORS['warning']}; 
        color: white; 
        padding: 0.25rem 0.75rem; 
        border-radius: 12px; 
        font-size: 0.75rem; 
        font-weight: 500;
    }}
    .distance-far {{ 
        background: {COLORS['danger']}; 
        color: white; 
        padding: 0.25rem 0.75rem; 
        border-radius: 12px; 
        font-size: 0.75rem; 
        font-weight: 500;
    }}
</style>
""", unsafe_allow_html=True)

# Helper function to calculate distance
def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    try:
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        
        return c * r
    except (ValueError, TypeError):
        return 0.0

# Gateway coordinates
def get_gateway_coordinates():
    """Define gateway locations"""
    return {
        'wp-gateway-1': {'lat': 3.1276, 'lon': 38.2772},
        'busia-gateway1-wirelessplanet': {'lat': -1.2632, 'lon': 36.7604},
        'wireless-planet-busia': {'lat': -1.2632, 'lon': 36.7604},
        'busia-gateway2-wirelessplanet': {'lat': -1.2620, 'lon': 36.7590}
    }

# Professional metric card function
def create_metric_card(title, value, delta=None, status="normal", icon="📊"):
    status_class = f"status-{status}" if status in ["excellent", "good", "poor"] else "metric-card"
    delta_html = ""
    if delta is not None:
        delta_color = COLORS['success'] if delta >= 0 else COLORS['danger']
        delta_html = f'<div class="metric-delta" style="color: {delta_color};">Δ {delta}</div>'
    
    return f"""
    <div class="{status_class} metric-card">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem; color: {COLORS['secondary']};">{icon}</div>
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

# Data cleaning function
def clean_data(df):
    """Clean and prepare data for analysis"""
    try:
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Clean numeric columns
        numeric_columns = ['rssi', 'snr', 'lat', 'lon', 'sf', 'f_cnt']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean string columns
        string_columns = ['device_id', 'gateway_id', 'device_type']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Remove rows with invalid timestamps
        df = df.dropna(subset=['timestamp'])
        
        # Create additional time columns
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        return df
    except Exception as e:
        st.error(f"Error cleaning data: {str(e)}")
        return df

# Function to analyze device-gateway connections
def analyze_device_connections(df, device_id, gateway_coords):
    """Analyze connections with proper data type handling"""
    try:
        device_data = df[df['device_id'].astype(str) == str(device_id)].copy()
        
        if len(device_data) == 0:
            return None
        
        # Get device location safely
        device_lat = float(device_data['lat'].iloc[0])
        device_lon = float(device_data['lon'].iloc[0])
        
        connections = []
        
        for _, row in device_data.iterrows():
            gateway_id = str(row['gateway_id'])
            
            if gateway_id in gateway_coords:
                gateway_lat = float(gateway_coords[gateway_id]['lat'])
                gateway_lon = float(gateway_coords[gateway_id]['lon'])
                distance = calculate_distance(device_lat, device_lon, gateway_lat, gateway_lon)
                
                connections.append({
                    'timestamp': row['timestamp'],
                    'gateway_id': gateway_id,
                    'rssi': float(row['rssi']) if pd.notna(row['rssi']) else -999,
                    'snr': float(row['snr']) if pd.notna(row['snr']) else 0,
                    'distance_km': distance,
                    'sf': int(row.get('sf', 7)) if pd.notna(row.get('sf', 7)) else 7
                })
        
        return pd.DataFrame(connections) if connections else None
    except Exception as e:
        st.error(f"Error analyzing connections: {str(e)}")
        return None

# Professional plotting functions
def create_professional_pie(labels, values, title, colors=None):
    """Create professional pie chart"""
    if colors is None:
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['danger']]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(
            colors=colors,
            line=dict(color=COLORS['white'], width=2)
        ),
        textinfo='label+percent',
        textfont=dict(size=11, color=COLORS['text']),
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14, color=COLORS['text'])),
        showlegend=True,
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5),
        margin=dict(t=50, b=50, l=50, r=50),
        paper_bgcolor=COLORS['white'],
        plot_bgcolor=COLORS['white'],
        height=350
    )
    
    return fig

def create_professional_bar(data, x_col, y_col, title, color=None):
    """Create professional bar chart"""
    fig = px.bar(
        data, x=x_col, y=y_col, 
        title=title,
        color_discrete_sequence=[color or COLORS['primary']]
    )
    
    fig.update_layout(
        title=dict(x=0.5, font=dict(size=14, color=COLORS['text'])),
        plot_bgcolor=COLORS['white'],
        paper_bgcolor=COLORS['white'],
        font=dict(color=COLORS['text']),
        height=350
    )
    
    fig.update_traces(marker_line_color=COLORS['white'], marker_line_width=1)
    return fig

def create_connection_timeline(connections_df):
    """Create professional connection timeline"""
    if connections_df is None or len(connections_df) == 0:
        return None
    
    # Professional color palette for gateways
    gateway_colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['warning']]
    
    fig = go.Figure()
    
    for i, gateway in enumerate(connections_df['gateway_id'].unique()):
        gateway_data = connections_df[connections_df['gateway_id'] == gateway]
        color = gateway_colors[i % len(gateway_colors)]
        
        fig.add_trace(go.Scatter(
            x=gateway_data['timestamp'],
            y=gateway_data['rssi'],
            mode='markers+lines',
            name=gateway,
            marker=dict(size=8, color=color, line=dict(width=1, color=COLORS['white'])),
            line=dict(width=2, color=color),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Time: %{x}<br>' +
                         'RSSI: %{y} dBm<br>' +
                         'Distance: %{customdata[0]:.2f} km<br>' +
                         'SNR: %{customdata[1]:.1f} dB<br>' +
                         '<extra></extra>',
            customdata=gateway_data[['distance_km', 'snr']].values
        ))
    
    fig.update_layout(
        title='Device Connection Timeline',
        xaxis_title='Time',
        yaxis_title='RSSI (dBm)',
        height=400,
        plot_bgcolor=COLORS['white'],
        paper_bgcolor=COLORS['white'],
        font=dict(color=COLORS['text'])
    )
    
    # Add professional threshold lines
    fig.add_hline(y=-70, line_dash="dash", line_color=COLORS['success'], 
                 annotation_text="Good Signal", annotation_position="top left")
    fig.add_hline(y=-90, line_dash="dash", line_color=COLORS['danger'], 
                 annotation_text="Poor Signal", annotation_position="bottom left")
    
    return fig

def create_network_map(df, selected_device=None):
    """Create professional network map"""
    gateway_coords = get_gateway_coordinates()
    
    fig = go.Figure()
    
    # Add gateways
    gateway_lats = [coords['lat'] for coords in gateway_coords.values()]
    gateway_lons = [coords['lon'] for coords in gateway_coords.values()]
    gateway_names = list(gateway_coords.keys())
    
    fig.add_trace(go.Scattermapbox(
        lat=gateway_lats,
        lon=gateway_lons,
        mode='markers',
        marker=dict(size=12, color=COLORS['primary']),
        text=gateway_names,
        name='Gateways',
        hovertemplate='<b>Gateway:</b> %{text}<br><b>Location:</b> %{lat:.4f}, %{lon:.4f}<extra></extra>'
    ))
    
    # Add devices
    device_locations = df.groupby(['device_id', 'device_type']).agg({
        'lat': 'first',
        'lon': 'first',
        'rssi': 'mean',
        'timestamp': 'count'
    }).reset_index()
    
    # Professional device colors
    device_colors = {
        'solar': COLORS['success'],
        'ac': COLORS['secondary'],
        'battery': COLORS['warning']
    }
    
    for device_type in device_locations['device_type'].unique():
        device_subset = device_locations[device_locations['device_type'] == device_type]
        
        fig.add_trace(go.Scattermapbox(
            lat=device_subset['lat'],
            lon=device_subset['lon'],
            mode='markers',
            marker=dict(size=8, color=device_colors.get(device_type, COLORS['accent'])),
            text=device_subset['device_id'],
            name=f'{device_type.title()} Devices',
            hovertemplate='<b>Device:</b> %{text}<br><b>Type:</b> ' + device_type + 
                         '<br><b>Messages:</b> %{customdata[0]:,}<br><b>Avg RSSI:</b> %{customdata[1]:.1f} dBm<extra></extra>',
            customdata=device_subset[['timestamp', 'rssi']].values
        ))
    
    # Highlight selected device
    if selected_device and selected_device != 'All Devices':
        selected_data = device_locations[device_locations['device_id'].astype(str) == str(selected_device)]
        if len(selected_data) > 0:
            fig.add_trace(go.Scattermapbox(
                lat=selected_data['lat'],
                lon=selected_data['lon'],
                mode='markers',
                marker=dict(size=15, color=COLORS['danger'], symbol='star'),
                text=selected_data['device_id'],
                name='Selected Device',
                hovertemplate='<b>Selected Device:</b> %{text}<extra></extra>'
            ))
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=df['lat'].mean(), lon=df['lon'].mean()),
            zoom=8
        ),
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor=COLORS['white']
    )
    
    return fig

# Main application
def main():
    # Professional header
    st.markdown(f'<h1 class="main-header">📡 LoRaWAN Network Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload LoRaWAN Data",
            type=['xlsx', 'xls', 'csv'],
            help="Upload your LoRaWAN network data file"
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner('Loading and processing data...'):
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    # Clean the data
                    df = clean_data(df)
                
                st.success(f"✅ {len(df):,} records loaded successfully!")
                
                # Professional filters
                st.markdown("### 🔍 Filters")
                
                device_list = ['All Devices'] + sorted([str(d) for d in df['device_id'].unique()])
                selected_device = st.selectbox("📱 Device", device_list)
                
                gateway_list = ['All Gateways'] + sorted(df['gateway_id'].unique())
                selected_gateway = st.selectbox("🌐 Gateway", gateway_list)
                
                signal_filter = st.selectbox(
                    "📶 Signal Quality",
                    ["All Signals", "Strong (>-70 dBm)", "Moderate (-70 to -90 dBm)", "Weak (<-90 dBm)"]
                )
                
                # Apply filters
                filtered_df = df.copy()
                
                if selected_device != 'All Devices':
                    filtered_df = filtered_df[filtered_df['device_id'].astype(str) == selected_device]
                
                if selected_gateway != 'All Gateways':
                    filtered_df = filtered_df[filtered_df['gateway_id'] == selected_gateway]
                
                if signal_filter != "All Signals":
                    if signal_filter == "Strong (>-70 dBm)":
                        filtered_df = filtered_df[filtered_df['rssi'] > -70]
                    elif signal_filter == "Moderate (-70 to -90 dBm)":
                        filtered_df = filtered_df[(filtered_df['rssi'] >= -90) & (filtered_df['rssi'] <= -70)]
                    elif signal_filter == "Weak (<-90 dBm)":
                        filtered_df = filtered_df[filtered_df['rssi'] < -90]
                
                if len(filtered_df) != len(df):
                    st.info(f"📊 {len(filtered_df):,} of {len(df):,} records")
                
                # Quick stats
                st.markdown("### 📈 Quick Stats")
                st.metric("Devices", filtered_df['device_id'].nunique())
                st.metric("Gateways", filtered_df['gateway_id'].nunique())
                st.metric("Avg RSSI", f"{filtered_df['rssi'].mean():.1f} dBm")
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                st.info("Please check your file format and column names.")
                return
    
    if uploaded_file is None:
        # Professional landing page
        st.markdown(f"""
        <div style="text-align: center; padding: 3rem 0; color: {COLORS['text']};">
            <h2 style="color: {COLORS['primary']};">Professional LoRaWAN Network Analysis</h2>
            <p style="font-size: 1.1rem; color: {COLORS['muted']}; margin-bottom: 3rem;">
                Enterprise-grade analytics for network optimization and performance monitoring
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card("Real-time Monitoring", "24/7", icon="📊"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Device Analytics", "Individual", icon="📱"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("Gateway Analysis", "Multi-Gateway", icon="🌐"), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("Distance Tracking", "Real-time", icon="📏"), unsafe_allow_html=True)
        
        st.markdown(f"""
        ### Key Features
        
        - **Device Connection Analysis**: Track gateway connections and signal quality
        - **Distance Calculations**: Automatic distance measurement between devices and gateways  
        - **Signal Quality Mapping**: Visualize performance vs distance relationships
        - **Professional Reports**: Generate comprehensive network analysis reports
        """)
        
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📱 Device Analysis", 
        "🗺️ Network Map",
        "🌐 Gateway Performance", 
        "📋 Reports"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.markdown("### Network Performance Overview")
        
        # Health assessment
        strong_pct = (filtered_df['rssi'] > -70).mean() * 100
        if strong_pct > 80:
            health_color = COLORS['success']
            health_message = "🟢 Excellent Network Performance"
        elif strong_pct > 60:
            health_color = COLORS['warning']
            health_message = "🟡 Good Network Performance"
        else:
            health_color = COLORS['danger']
            health_message = "🔴 Network Needs Optimization"
        
        st.markdown(f"""
        <div class="health-indicator" style="background: {health_color};">
            {health_message}
        </div>
        """, unsafe_allow_html=True)
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card("Total Messages", f"{len(filtered_df):,}", icon="📨"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Active Devices", filtered_df['device_id'].nunique(), icon="📱"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("Active Gateways", filtered_df['gateway_id'].nunique(), icon="🌐"), unsafe_allow_html=True)
        with col4:
            avg_rssi = filtered_df['rssi'].mean()
            status = "excellent" if avg_rssi > -70 else "good" if avg_rssi > -85 else "poor"
            st.markdown(create_metric_card("Signal Quality", f"{avg_rssi:.1f} dBm", status=status, icon="📶"), unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Signal quality distribution
            strong = (filtered_df['rssi'] > -70).sum()
            moderate = ((filtered_df['rssi'] >= -90) & (filtered_df['rssi'] <= -70)).sum()
            weak = (filtered_df['rssi'] < -90).sum()
            
            fig_signal = create_professional_pie(
                ['Strong (>-70 dBm)', 'Moderate (-70 to -90 dBm)', 'Weak (<-90 dBm)'],
                [strong, moderate, weak],
                "Signal Quality Distribution",
                [COLORS['success'], COLORS['warning'], COLORS['danger']]
            )
            st.plotly_chart(fig_signal, use_container_width=True)
        
        with col2:
            # Gateway load
            gateway_counts = filtered_df['gateway_id'].value_counts().reset_index()
            gateway_counts.columns = ['gateway_id', 'messages']
            
            fig_gateway = create_professional_bar(
                gateway_counts, 'gateway_id', 'messages', 
                "Gateway Load Distribution", COLORS['primary']
            )
            st.plotly_chart(fig_gateway, use_container_width=True)
        
        # Activity timeline
        daily_activity = filtered_df.groupby(filtered_df['timestamp'].dt.date).size().reset_index()
        daily_activity.columns = ['date', 'messages']
        
        fig_timeline = px.line(
            daily_activity, x='date', y='messages',
            title="Daily Network Activity",
            color_discrete_sequence=[COLORS['primary']]
        )
        fig_timeline.update_layout(
            plot_bgcolor=COLORS['white'],
            paper_bgcolor=COLORS['white'],
            font=dict(color=COLORS['text'])
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Tab 2: Device Analysis
    with tab2:
        st.markdown("### Device Performance Analysis")
        
        if selected_device == 'All Devices':
            st.info("Select a specific device from the sidebar for detailed analysis")
            
            # Device summary
            device_summary = filtered_df.groupby(['device_id', 'device_type']).agg({
                'timestamp': 'count',
                'rssi': 'mean',
                'snr': 'mean',
                'gateway_id': 'nunique'
            }).round(2)
            device_summary.columns = ['messages', 'avg_rssi', 'avg_snr', 'gateways']
            device_summary.reset_index(inplace=True)
            
            st.dataframe(device_summary, use_container_width=True)
            
        else:
            device_data = filtered_df[filtered_df['device_id'].astype(str) == selected_device]
            
            if len(device_data) == 0:
                st.error("No data found for selected device")
                return
            
            # Device header
            device_type = device_data['device_type'].iloc[0]
            st.markdown(f"""
            <div class="info-box">
                <h3>📱 Device {selected_device} ({device_type})</h3>
                <p><strong>Location:</strong> {device_data['lat'].iloc[0]:.4f}, {device_data['lon'].iloc[0]:.4f}</p>
                <p><strong>Period:</strong> {device_data['timestamp'].min()} to {device_data['timestamp'].max()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Device KPIs
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(create_metric_card("Messages", f"{len(device_data):,}", icon="📨"), unsafe_allow_html=True)
            
            with col2:
                avg_rssi_device = device_data['rssi'].mean()
                status = "excellent" if avg_rssi_device > -70 else "good" if avg_rssi_device > -85 else "poor"
                st.markdown(create_metric_card("Avg RSSI", f"{avg_rssi_device:.1f} dBm", status=status, icon="📶"), unsafe_allow_html=True)
            
            with col3:
                st.markdown(create_metric_card("Avg SNR", f"{device_data['snr'].mean():.1f} dB", icon="📡"), unsafe_allow_html=True)
            
            with col4:
                st.markdown(create_metric_card("Gateways", device_data['gateway_id'].nunique(), icon="🌐"), unsafe_allow_html=True)
            
            # Gateway connections analysis
            st.markdown("#### Gateway Connection Analysis")
            
            gateway_coords = get_gateway_coordinates()
            device_lat = float(device_data['lat'].iloc[0])
            device_lon = float(device_data['lon'].iloc[0])
            
            gateway_analysis = []
            for gateway_id in device_data['gateway_id'].unique():
                gw_data = device_data[device_data['gateway_id'] == gateway_id]
                
                if str(gateway_id) in gateway_coords:
                    gw_coords = gateway_coords[str(gateway_id)]
                    distance = calculate_distance(device_lat, device_lon, gw_coords['lat'], gw_coords['lon'])
                    
                    # Distance category
                    if distance < 1:
                        distance_category, distance_label = "close", "Close"
                    elif distance < 5:
                        distance_category, distance_label = "medium", "Medium"
                    else:
                        distance_category, distance_label = "far", "Far"
                    
                    gateway_analysis.append({
                        'gateway_id': str(gateway_id),
                        'messages': len(gw_data),
                        'avg_rssi': gw_data['rssi'].mean(),
                        'avg_snr': gw_data['snr'].mean(),
                        'distance_km': distance,
                        'distance_category': distance_category,
                        'distance_label': distance_label,
                        'connection_pct': len(gw_data) / len(device_data) * 100
                    })
            
            # Display connections
            for gw in sorted(gateway_analysis, key=lambda x: x['messages'], reverse=True):
                st.markdown(f"""
                <div class="gateway-connection">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <h4 style="margin: 0; color: {COLORS['primary']};">🌐 {gw['gateway_id']}</h4>
                        <span class="distance-{gw['distance_category']}">{gw['distance_km']:.2f} km</span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem; font-size: 0.9rem;">
                        <div><strong>Messages:</strong> {gw['messages']:,}</div>
                        <div><strong>Usage:</strong> {gw['connection_pct']:.1f}%</div>
                        <div><strong>RSSI:</strong> {gw['avg_rssi']:.1f} dBm</div>
                        <div><strong>SNR:</strong> {gw['avg_snr']:.1f} dB</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Connection timeline
            st.markdown("#### Connection Timeline")
            connections_df = analyze_device_connections(device_data, selected_device, gateway_coords)
            
            if connections_df is not None and len(connections_df) > 0:
                timeline_fig = create_connection_timeline(connections_df)
                if timeline_fig:
                    st.plotly_chart(timeline_fig, use_container_width=True)
                
                # Analysis charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distance vs Signal plot
                    fig_distance = px.scatter(
                        connections_df, x='distance_km', y='rssi', color='gateway_id',
                        title='Distance vs Signal Quality',
                        labels={'distance_km': 'Distance (km)', 'rssi': 'RSSI (dBm)'},
                        color_discrete_sequence=[COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['warning']]
                    )
                    fig_distance.update_layout(
                        plot_bgcolor=COLORS['white'],
                        paper_bgcolor=COLORS['white'],
                        font=dict(color=COLORS['text']),
                        height=350
                    )
                    st.plotly_chart(fig_distance, use_container_width=True)
                
                with col2:
                    # Gateway usage
                    usage_data = connections_df['gateway_id'].value_counts()
                    fig_usage = create_professional_pie(
                        usage_data.index, usage_data.values,
                        "Gateway Usage Distribution"
                    )
                    st.plotly_chart(fig_usage, use_container_width=True)
    
    # Tab 3: Network Map
    with tab3:
        st.markdown("### Network Geographic View")
        
        try:
            network_fig = create_network_map(filtered_df, selected_device if selected_device != 'All Devices' else None)
            st.plotly_chart(network_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating map: {str(e)}")
            st.info("Map functionality requires valid coordinate data.")
        
        # Coverage stats
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            #### Map Legend
            - **Blue Markers**: Gateway locations
            - **Green/Gray Circles**: Device locations by type
            - **Red Star**: Selected device (if any)
            """)
        
        with col2:
            if len(filtered_df) > 0:
                gateway_coords = get_gateway_coordinates()
                device_locations = filtered_df[['device_id', 'lat', 'lon']].drop_duplicates()
                
                coverage_area = (filtered_df['lat'].max() - filtered_df['lat'].min()) * (filtered_df['lon'].max() - filtered_df['lon'].min())
                
                st.markdown(f"""
                #### Coverage Statistics
                - **Devices**: {len(device_locations)}
                - **Gateways**: {len(gateway_coords)}
                - **Coverage Area**: {coverage_area:.6f} sq°
                """)
    
    # Tab 4: Gateway Performance
    with tab4:
        st.markdown("### Gateway Performance Analysis")
        
        # Gateway stats
        gateway_stats = filtered_df.groupby('gateway_id').agg({
            'rssi': ['mean', 'std', 'count'],
            'snr': 'mean',
            'device_id': 'nunique'
        }).round(2)
        
        gateway_stats.columns = ['avg_rssi', 'rssi_std', 'messages', 'avg_snr', 'devices']
        gateway_stats.reset_index(inplace=True)
        
        # Performance score
        gateway_stats['performance_score'] = ((gateway_stats['avg_rssi'] + 100) * gateway_stats['avg_snr'] / 10).round(1)
        
        st.dataframe(gateway_stats, use_container_width=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_box = px.box(
                filtered_df, x='gateway_id', y='rssi',
                title="RSSI Distribution by Gateway",
                color_discrete_sequence=[COLORS['primary']]
            )
            fig_box.add_hline(y=-70, line_dash="dash", line_color=COLORS['success'], annotation_text="Good")
            fig_box.add_hline(y=-90, line_dash="dash", line_color=COLORS['danger'], annotation_text="Poor")
            fig_box.update_layout(
                plot_bgcolor=COLORS['white'],
                paper_bgcolor=COLORS['white'],
                font=dict(color=COLORS['text']),
                height=350
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            fig_performance = create_professional_bar(
                gateway_stats, 'gateway_id', 'performance_score',
                "Gateway Performance Score", COLORS['secondary']
            )
            st.plotly_chart(fig_performance, use_container_width=True)
        
        # Load balancing
        st.markdown("#### Load Balancing Analysis")
        
        gateway_load = filtered_df['gateway_id'].value_counts()
        if len(gateway_load) > 1:
            balance_ratio = gateway_load.max() / gateway_load.min()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Balance Ratio", f"{balance_ratio:.2f}")
            with col2:
                st.metric("Highest Load", f"{gateway_load.max():,}")
            with col3:
                st.metric("Lowest Load", f"{gateway_load.min():,}")
            
            if balance_ratio > 3:
                st.warning("⚠️ Significant load imbalance detected")
            elif balance_ratio > 2:
                st.info("ℹ️ Moderate load imbalance")
            else:
                st.success("✅ Well-balanced load distribution")
    
    # Tab 5: Reports
    with tab5:
        st.markdown("### Analysis Reports")
        
        # Generate insights
        insights = []
        
        # Signal quality
        strong_pct = (filtered_df['rssi'] > -70).mean() * 100
        if strong_pct > 80:
            insights.append("✅ Excellent signal quality - over 80% strong signals")
        elif strong_pct > 60:
            insights.append("⚠️ Good signal quality with optimization opportunities")
        else:
            insights.append("❌ Poor signal quality - immediate attention required")
        
        # Gateway load
        if filtered_df['gateway_id'].nunique() > 1:
            gateway_counts = filtered_df['gateway_id'].value_counts()
            load_ratio = gateway_counts.max() / gateway_counts.min()
            if load_ratio < 2:
                insights.append("✅ Well-balanced gateway distribution")
            else:
                insights.append("⚠️ Gateway load imbalance detected")
        
        # Device connectivity
        if selected_device != 'All Devices':
            device_data = filtered_df[filtered_df['device_id'].astype(str) == selected_device]
            if len(device_data) > 0:
                gw_count = device_data['gateway_id'].nunique()
                if gw_count > 2:
                    insights.append(f"✅ Device {selected_device} has excellent connectivity")
                elif gw_count > 1:
                    insights.append(f"⚠️ Device {selected_device} has moderate connectivity")
                else:
                    insights.append(f"❌ Device {selected_device} has limited connectivity")
        
        # Display insights
        st.markdown("#### Key Insights")
        for insight in insights:
            st.markdown(f"• {insight}")
        
        # Recommendations
        st.markdown("#### Recommendations")
        recommendations = [
            "Monitor devices with weak signals for optimization opportunities",
            "Consider load balancing for better gateway utilization",
            "Implement regular network health monitoring",
            "Review device placement for optimal coverage"
        ]
        
        for rec in recommendations:
            st.markdown(f"• {rec}")
        
        # Export
        st.markdown("#### Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download Data", use_container_width=True):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "💾 Download CSV",
                    csv,
                    f"lorawan_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        with col2:
            if st.button("📋 Generate Report", use_container_width=True):
                report = f"""
LoRaWAN Network Analysis Report
==============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
================
Total Messages: {len(filtered_df):,}
Active Devices: {filtered_df['device_id'].nunique()}
Active Gateways: {filtered_df['gateway_id'].nunique()}
Signal Quality: {strong_pct:.1f}% strong signals

KEY INSIGHTS
============
{chr(10).join([f'• {insight}' for insight in insights])}

RECOMMENDATIONS
===============
{chr(10).join([f'• {rec}' for rec in recommendations])}
                """
                
                st.download_button(
                    "💾 Download Report",
                    report,
                    f"lorawan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()