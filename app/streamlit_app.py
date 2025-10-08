"""
Smart Post-Purchase AI Guardian - Main Streamlit App
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Import components
from components.chat_interface import render_chat_interface
from components.analytics_dashboard import render_analytics_dashboard
from components.visual_upload import render_visual_upload

# Page config
st.set_page_config(
    page_title="AI Guardian - E-commerce Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS with fixes
def load_css():
    """Load custom CSS with proper visibility"""
    st.markdown("""
    <style>
    /* Fix text visibility */
    .stMarkdown, .stText, p, span, div {
        color: #FFFFFF !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1E1E1E;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #FFFFFF !important;
    }
    
    /* Main container */
    .main {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    
    .stButton > button:hover {
        background-color: #FF6B6B;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #FFFFFF;
        border: 1px solid #4A4A4A;
    }
    
    /* Success/Error boxes */
    .stSuccess {
        background-color: #1B5E20;
        color: #FFFFFF;
    }
    
    .stError {
        background-color: #B71C1C;
        color: #FFFFFF;
    }
    
    .stInfo {
        background-color: #0D47A1;
        color: #FFFFFF;
    }
    
    /* Radio buttons - fix labels */
    .stRadio > label {
        color: #FFFFFF !important;
    }
    
    .stRadio > div {
        color: #FFFFFF !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = f"CONV_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'agents_initialized' not in st.session_state:
        st.session_state.agents_initialized = False
    
    if 'initialization_error' not in st.session_state:
        st.session_state.initialization_error = None
    
    if 'order_id' not in st.session_state:
        st.session_state.order_id = None
    
    if 'customer_name' not in st.session_state:
        st.session_state.customer_name = "Customer"

# Initialize agents
def initialize_agents():
    """Initialize all agents with API keys"""
    if st.session_state.agents_initialized:
        return True
    
    try:
        openai_key = os.getenv('OPENAI_API_KEY')
        gemini_key = os.getenv('GEMINI_API_KEY')
        pinecone_key = os.getenv('PINECONE_API_KEY')
        
        if not openai_key or not pinecone_key:
            st.session_state.initialization_error = "Missing required API keys in .env file"
            return False
        
        from agents import ControllerAgent, MonitorAgent, VisualAgent, ExchangeAgent, ResolutionAgent
        
        config = {
            'openai_api_key': openai_key,
            'gemini_api_key': gemini_key or 'not-set',
            'pinecone_api_key': pinecone_key,
            'pinecone_index_name': os.getenv('PINECONE_INDEX_NAME', 'ecommerce-guardian')
        }
        
        st.session_state.controller = ControllerAgent(config)
        st.session_state.monitor = MonitorAgent(config)
        st.session_state.visual = VisualAgent(config) if gemini_key else None
        st.session_state.exchange = ExchangeAgent(config)
        st.session_state.resolution = ResolutionAgent(config)
        
        st.session_state.agents_initialized = True
        st.session_state.initialization_error = None
        
        return True
        
    except Exception as e:
        st.session_state.initialization_error = str(e)
        return False

# Main app
def main():
    """Main application"""
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 AI Guardian")
        st.markdown("### Smart Post-Purchase Support")
        st.divider()
        
        if not st.session_state.agents_initialized:
            with st.spinner("Initializing..."):
                success = initialize_agents()
            
            if not success:
                st.error("⚠️ Initialization Failed")
                with st.expander("Error Details"):
                    st.text(st.session_state.initialization_error)
        else:
            st.success("✅ System Active")
        
        st.markdown("#### Customer Information")
        st.session_state.customer_name = st.text_input(
            "Customer Name", 
            value=st.session_state.customer_name,
            key="cust_name"
        )
        st.session_state.order_id = st.text_input(
            "Order ID (Optional)", 
            value=st.session_state.order_id or "",
            placeholder="ORD123456",
            key="order_id_input"
        )
        
        st.divider()
        
        st.markdown("#### Navigation")
        # Fixed radio button with proper label
        page = st.radio(
            label="Choose a page",  # Fixed: Added proper label
            options=["💬 Chat Interface", "📊 Analytics Dashboard", "📸 Visual Upload"],
            label_visibility="hidden"  # Hide the label visually
        )
        
        st.divider()
        
        with st.expander("ℹ️ System Info"):
            st.write(f"Conversation: {st.session_state.conversation_id[:15]}...")
            st.write(f"Messages: {len(st.session_state.messages)}")
        
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = f"CONV_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.rerun()
    
    # Main content
    if st.session_state.initialization_error and not st.session_state.agents_initialized:
        st.error("⚠️ Configuration Required")
        st.markdown("""
        ### Setup Instructions:
        
        Create a `.env` file in the project root with:
        ```
        OPENAI_API_KEY=your-openai-api-key
        PINECONE_API_KEY=your-pinecone-api-key
        PINECONE_INDEX_NAME=ecommerce-guardian
        GEMINI_API_KEY=your-gemini-api-key
        ```
        """)
        st.stop()
    
    # Render pages
    if page == "💬 Chat Interface":
        render_chat_interface()
    elif page == "📊 Analytics Dashboard":
        render_analytics_dashboard()
    elif page == "📸 Visual Upload":
        render_visual_upload()

if __name__ == "__main__":
    main()
