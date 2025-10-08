"""
Analytics Dashboard Component
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def render_analytics_dashboard():
    """Render analytics dashboard"""
    
    st.title("📊 Analytics Dashboard")
    st.markdown("Real-time insights into AI agent performance")
    
    # Generate demo data
    metrics_data = generate_demo_metrics()
    
    # KPI Metrics
    st.markdown("### Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Conversations",
            value=metrics_data['total_conversations'],
            delta="+23 today"
        )
    
    with col2:
        st.metric(
            label="Resolution Rate",
            value=f"{metrics_data['resolution_rate']}%",
            delta="+5.2%"
        )
    
    with col3:
        st.metric(
            label="Avg Response Time",
            value=f"{metrics_data['avg_response_time']}s",
            delta="-2.1s"
        )
    
    with col4:
        st.metric(
            label="Customer Satisfaction",
            value=f"{metrics_data['satisfaction_score']}/5.0",
            delta="+0.3"
        )
    
    st.divider()
    
    # Agent Performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Agent Usage Distribution")
        agent_data = pd.DataFrame({
            'Agent': ['Controller', 'Monitor', 'Exchange', 'Resolution', 'Visual'],
            'Requests': [450, 320, 180, 150, 90]
        })
        st.bar_chart(agent_data.set_index('Agent'))
    
    with col2:
        st.markdown("### Issue Types")
        issue_data = pd.DataFrame({
            'Issue': ['Tracking', 'Size Exchange', 'Refund', 'Defect', 'Other'],
            'Count': [280, 190, 150, 85, 45]
        })
        st.bar_chart(issue_data.set_index('Issue'))
    
    st.divider()
    
    # Recent conversations
    st.markdown("### Recent Conversations")
    
    recent_convs = pd.DataFrame({
        'Time': ['5 min ago', '12 min ago', '25 min ago', '1 hour ago', '2 hours ago'],
        'Customer': ['John D.', 'Sarah M.', 'Mike R.', 'Emily K.', 'David L.'],
        'Issue': ['Order Tracking', 'Size Exchange', 'Refund Request', 'Defect Report', 'Color Exchange'],
        'Agent': ['Monitor', 'Exchange', 'Resolution', 'Visual', 'Exchange'],
        'Status': ['✅ Resolved', '✅ Resolved', '⏳ In Progress', '✅ Resolved', '✅ Resolved'],
        'Satisfaction': ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '-', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐']
    })
    
    st.dataframe(recent_convs, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Cost savings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Cost Savings")
        st.info("**AI vs Human Agent Cost**")
        st.write("- AI Resolution: $0.15/conversation")
        st.write("- Human Resolution: $8.50/conversation")
        st.write("- **Monthly Savings:** $12,450")
    
    with col2:
        st.markdown("### Return Prevention")
        st.success("**Returns Prevented This Month**")
        st.write("- Proactive interventions: 145")
        st.write("- Returns prevented: 98 (67.6%)")
        st.write("- **Revenue Retained:** $8,234")

def generate_demo_metrics():
    """Generate demo metrics data"""
    return {
        'total_conversations': 1247,
        'resolution_rate': 87.3,
        'avg_response_time': 4.8,
        'satisfaction_score': 4.6
    }
