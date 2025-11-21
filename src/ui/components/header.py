"""
Header Component
"""
import streamlit as st

def render():
    """Render the application header and load global CSS"""
    
    # Custom CSS for "Deep Ocean" UI styling
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&family=Roboto+Mono:wght@400;500&display=swap');

    /* Global Font Settings */
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }
    
    /* App background - Deep Ocean Gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #050A14 0%, #0A192F 50%, #020C1B 100%);
        color: #E6F1FF !important;
    }
    
    /* Header - Transparent */
    [data-testid="stHeader"] {
        background: rgba(5, 10, 20, 0.8) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar - Glassmorphism */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 25, 47, 0.95) !important;
        border-right: 1px solid rgba(100, 255, 218, 0.1);
    }

    /* Headings */
    h1, h2, h3 {
        color: #64FFDA !important; /* Neon Teal */
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    .main-header {
        font-family: 'Montserrat', sans-serif;
        font-size: 3rem;
        background: linear-gradient(90deg, #64FFDA, #00B4D8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        text-shadow: 0 0 20px rgba(100, 255, 218, 0.3);
    }

    /* Buttons - Neon Glow */
    .stButton>button {
        background: rgba(100, 255, 218, 0.1) !important;
        color: #64FFDA !important;
        border: 1px solid #64FFDA !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9rem !important;
    }
    .stButton>button:hover {
        background: rgba(100, 255, 218, 0.2) !important;
        box-shadow: 0 0 15px rgba(100, 255, 218, 0.4) !important;
        transform: translateY(-2px);
    }
    
    /* Primary Button */
    .stButton>button[kind="primary"] {
        background: linear-gradient(45deg, #64FFDA, #00B4D8) !important;
        color: #020C1B !important;
        border: none !important;
    }
    .stButton>button[kind="primary"]:hover {
        box-shadow: 0 0 20px rgba(100, 255, 218, 0.6) !important;
    }

    /* Inputs & Selects */
    .stSelectbox, .stTextInput, .stNumberInput, .stTextArea {
        color: #E6F1FF !important;
    }
    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(100, 255, 218, 0.2) !important;
        color: #E6F1FF !important;
    }
    div[data-baseweb="input"] > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(100, 255, 218, 0.2) !important;
        color: #E6F1FF !important;
    }

    /* Cards/Containers */
    div[data-testid="stExpander"] {
        background-color: rgba(17, 34, 64, 0.6) !important;
        border: 1px solid rgba(100, 255, 218, 0.1) !important;
        border-radius: 10px !important;
    }

    /* Info boxes with glassmorphism */
    .success-box {
        background: rgba(27, 94, 32, 0.2);
        backdrop-filter: blur(5px);
        border-left: 4px solid #64FFDA;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
        color: #E6F1FF;
    }
    .info-box {
        background: rgba(1, 87, 155, 0.2);
        backdrop-filter: blur(5px);
        border-left: 4px solid #00B4D8;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
        color: #E6F1FF;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #64FFDA !important;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 4px 4px 0 0;
        color: #8892b0;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(100, 255, 218, 0.1) !important;
        color: #64FFDA !important;
        border-bottom: 2px solid #64FFDA !important;
    }

    /* Sidebar Navigation Styling */
    section[data-testid="stSidebar"] .stRadio {
        background-color: transparent !important;
    }
    
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
        gap: 10px;
    }

    section[data-testid="stSidebar"] .stRadio label {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(100, 255, 218, 0.1);
        border-radius: 8px;
        padding: 12px 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        color: #8892b0 !important;
        font-family: 'Montserrat', sans-serif;
        font-weight: 500;
        display: flex;
        align-items: center;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(100, 255, 218, 0.1);
        border-color: #64FFDA;
        color: #64FFDA !important;
        transform: translateX(5px);
    }

    /* Selected State */
    section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        background: linear-gradient(90deg, rgba(100, 255, 218, 0.2), transparent);
        border-left: 4px solid #64FFDA;
        border-color: rgba(100, 255, 218, 0.2);
        color: #64FFDA !important;
    }
    
    /* Hide the actual radio circle */
    section[data-testid="stSidebar"] .stRadio div[role="radio"] div:first-child {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">eDNA Biodiversity Assessment System</h1>', unsafe_allow_html=True)
    st.markdown("---")
