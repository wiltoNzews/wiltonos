"""
WiltonOS Streamlit Enhancements
Custom Streamlit extensions for improved UI/UX in WiltonOS
"""

import streamlit as st
import numpy as np
import pandas as pd
import base64
from typing import Dict, Any, List, Optional, Union, Callable
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [ST_ENHANCEMENTS] %(message)s",
    handlers=[
        logging.FileHandler("logs/streamlit_enhancements.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("streamlit_enhancements")

# Constants for styling
GLIFO_COLOR = "#6E44FF"
COHERENCE_BG = "#0E1117"
COHERENCE_TEXT = "#E0E0E0"
COHERENCE_ACCENT = "#64B5F6"
COHERENCE_SUCCESS = "#66BB6A"
COHERENCE_WARNING = "#FFB74D"
COHERENCE_ERROR = "#EF5350"

def custom_css():
    """Apply custom CSS styling for WiltonOS"""
    st.markdown(f"""
    <style>
        /* Base theme overrides */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        
        /* Glifo + Coherence theme */
        .st-emotion-cache-18ni7ap {{
            background-color: {COHERENCE_BG};
            color: {COHERENCE_TEXT};
        }}
        
        /* Custom component classes */
        .wiltonos-card {{
            background-color: rgba(30, 40, 50, 0.8);
            border-left: 4px solid {GLIFO_COLOR};
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .wiltonos-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }}
        
        .wiltonos-header {{
            color: {COHERENCE_TEXT};
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }}
        
        .wiltonos-label {{
            font-size: 0.8rem;
            color: rgba(255,255,255,0.6);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.25rem;
        }}
        
        .wiltonos-value {{
            font-size: 1.2rem;
            font-weight: 500;
            color: {COHERENCE_TEXT};
        }}
        
        .wiltonos-code {{
            font-family: 'Fira Code', monospace;
            background-color: rgba(0,0,0,0.3);
            border-radius: 4px;
            padding: 1rem;
            color: {COHERENCE_TEXT};
            overflow-x: auto;
        }}
        
        .wiltonos-badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-right: 0.5rem;
        }}
        
        .wiltonos-badge-primary {{
            background-color: {GLIFO_COLOR};
            color: white;
        }}
        
        .wiltonos-badge-secondary {{
            background-color: {COHERENCE_ACCENT};
            color: white;
        }}
        
        .wiltonos-badge-success {{
            background-color: {COHERENCE_SUCCESS};
            color: white;
        }}
        
        .wiltonos-badge-warning {{
            background-color: {COHERENCE_WARNING};
            color: white;
        }}
        
        .wiltonos-badge-error {{
            background-color: {COHERENCE_ERROR};
            color: white;
        }}
        
        /* Animation classes */
        .pulse {{
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0% {{ opacity: 0.6; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0.6; }}
        }}
        
        .glow {{
            box-shadow: 0 0 10px {GLIFO_COLOR};
            animation: glow 2s infinite alternate;
        }}
        
        @keyframes glow {{
            from {{ box-shadow: 0 0 10px {GLIFO_COLOR}; }}
            to {{ box-shadow: 0 0 20px {GLIFO_COLOR}, 0 0 30px {COHERENCE_ACCENT}; }}
        }}
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: rgba(0,0,0,0.1);
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: rgba(110, 68, 255, 0.5);
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(110, 68, 255, 0.8);
        }}
    </style>
    """, unsafe_allow_html=True)

def card(title: str, content: str, badge: Optional[str] = None, badge_type: str = "primary"):
    """Display a styled card with title and content"""
    badge_html = ""
    if badge:
        badge_html = f'<span class="wiltonos-badge wiltonos-badge-{badge_type}">{badge}</span>'
    
    st.markdown(f"""
    <div class="wiltonos-card">
        <div class="wiltonos-header">{badge_html} {title}</div>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)

def metric_card(label: str, value: Union[str, int, float], delta: Optional[Union[str, int, float]] = None,
              prefix: str = "", suffix: str = ""):
    """Display a metric with label, value and optional delta"""
    delta_html = ""
    if delta is not None:
        delta_class = "success" if float(delta) >= 0 else "error"
        delta_symbol = "↑" if float(delta) >= 0 else "↓"
        delta_html = f'<div style="color: var(--{delta_class}-color);">{delta_symbol} {abs(float(delta))}</div>'
    
    st.markdown(f"""
    <div class="wiltonos-card">
        <div class="wiltonos-label">{label}</div>
        <div class="wiltonos-value">{prefix}{value}{suffix}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def code_block(code: str, language: str = "python"):
    """Display a code block with syntax highlighting"""
    st.markdown(f"""
    <div class="wiltonos-code">
        <pre><code class="{language}">{code}</code></pre>
    </div>
    """, unsafe_allow_html=True)

def glow_text(text: str, color: Optional[str] = None):
    """Display text with a glow effect"""
    color_style = f"color: {color};" if color else ""
    st.markdown(f"""
    <div style="{color_style} text-shadow: 0 0 10px currentColor; animation: glow 2s infinite alternate;">
        {text}
    </div>
    """, unsafe_allow_html=True)

def animated_counter(end_value: int, start_value: int = 0, duration: int = 1000,
                    prefix: str = "", suffix: str = ""):
    """Display an animated counter that counts up to a value"""
    st.markdown(f"""
    <div class="animated-counter" id="counter-{hash(str(end_value))}">
        {prefix}<span class="count">{start_value}</span>{suffix}
    </div>
    
    <script>
        // Animation function
        const animateCounter = (id, start, end, duration) => {{
            const counter = document.getElementById(id);
            if (!counter) return;
            
            const countDisplay = counter.querySelector('.count');
            let startTime = null;
            
            function updateCounter(timestamp) {{
                if (!startTime) startTime = timestamp;
                const progress = timestamp - startTime;
                const percentage = Math.min(progress / duration, 1);
                
                const currentCount = Math.floor(start + percentage * (end - start));
                countDisplay.textContent = currentCount.toLocaleString();
                
                if (percentage < 1) {{
                    window.requestAnimationFrame(updateCounter);
                }}
            }}
            
            window.requestAnimationFrame(updateCounter);
        }};
        
        // Run animation
        animateCounter('counter-{hash(str(end_value))}', {start_value}, {end_value}, {duration});
    </script>
    """, unsafe_allow_html=True)

def progress_ring(value: float, max_value: float = 100, size: int = 120, thickness: int = 8,
                color: str = GLIFO_COLOR, background_color: str = "rgba(255,255,255,0.1)",
                show_text: bool = True):
    """Display a circular progress indicator"""
    percentage = min(100, max(0, (value / max_value) * 100))
    
    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center;">
        <div class="progress-ring" style="position: relative; width: {size}px; height: {size}px;">
            <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
                <!-- Background circle -->
                <circle
                    cx="{size/2}"
                    cy="{size/2}"
                    r="{(size - thickness) / 2}"
                    fill="transparent"
                    stroke="{background_color}"
                    stroke-width="{thickness}"
                />
                
                <!-- Progress circle -->
                <circle
                    cx="{size/2}"
                    cy="{size/2}"
                    r="{(size - thickness) / 2}"
                    fill="transparent"
                    stroke="{color}"
                    stroke-width="{thickness}"
                    stroke-dasharray="{np.pi * (size - thickness)}"
                    stroke-dashoffset="{np.pi * (size - thickness) * (1 - percentage / 100)}"
                    transform="rotate(-90 {size/2} {size/2})"
                />
                
                {f'<text x="{size/2}" y="{size/2 + 5}" text-anchor="middle" fill="white" font-size="{size/5}px">{int(percentage)}%</text>' if show_text else ''}
            </svg>
        </div>
    </div>
    """, unsafe_allow_html=True)

def timeline(events: List[Dict[str, Any]]):
    """Display a vertical timeline of events"""
    timeline_html = ""
    
    for i, event in enumerate(events):
        date = event.get("date", "")
        title = event.get("title", "")
        description = event.get("description", "")
        icon = event.get("icon", "●")
        color = event.get("color", GLIFO_COLOR)
        
        is_last = i == len(events) - 1
        
        timeline_html += f"""
        <div style="display: flex; margin-bottom: {0 if is_last else '20px'};">
            <div style="position: relative; width: 40px; display: flex; justify-content: center;">
                <div style="
                    width: 20px;
                    height: 20px;
                    background-color: {color};
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 12px;
                    z-index: 2;
                ">{icon}</div>
                
                {'' if is_last else f'<div style="position: absolute; top: 20px; bottom: 0; width: 2px; background-color: rgba(255,255,255,0.1);"></div>'}
            </div>
            
            <div style="flex: 1; padding-left: 15px;">
                <div style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">{date}</div>
                <div style="font-weight: 500; margin: 4px 0;">{title}</div>
                <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">{description}</div>
            </div>
        </div>
        """
    
    st.markdown(f"""
    <div class="wiltonos-card">
        <div class="wiltonos-header">Timeline</div>
        <div style="padding: 10px 0;">
            {timeline_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def data_card(df: pd.DataFrame, max_rows: int = 5, title: str = "Data Overview"):
    """Display a dataframe in a stylized card with pagination"""
    # Convert the dataframe to HTML
    df_html = df.head(max_rows).to_html(index=False, classes=["dataframe-table"])
    total_rows = len(df)
    
    st.markdown(f"""
    <div class="wiltonos-card">
        <div class="wiltonos-header">{title}</div>
        <div style="font-size: 0.8rem; margin-bottom: 10px; color: rgba(255,255,255,0.6);">
            Showing {min(max_rows, total_rows)} of {total_rows} rows
        </div>
        <div style="overflow-x: auto;">
            {df_html}
        </div>
    </div>
    
    <style>
        .dataframe-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        .dataframe-table th {{
            background-color: rgba(110, 68, 255, 0.2);
            padding: 8px 12px;
            text-align: left;
            color: {COHERENCE_TEXT};
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .dataframe-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        
        .dataframe-table tr:hover td {{
            background-color: rgba(255,255,255,0.05);
        }}
    </style>
    """, unsafe_allow_html=True)

def pulsing_dot(size: int = 10, color: str = GLIFO_COLOR):
    """Display a pulsing dot, useful for indicating active status"""
    st.markdown(f"""
    <div style="
        display: inline-block;
        width: {size}px;
        height: {size}px;
        background-color: {color};
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    "></div>
    """, unsafe_allow_html=True)

def terminal(content: str, title: str = "Terminal", height: str = "300px"):
    """Display a terminal-like text box"""
    st.markdown(f"""
    <div style="
        background-color: #1a1a1a;
        border-radius: 8px;
        padding: 8px;
        margin-bottom: 16px;
        font-family: 'Fira Code', monospace;
    ">
        <div style="
            background-color: #333;
            padding: 6px 12px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            display: flex;
            align-items: center;
            font-size: 0.9rem;
        ">
            <div style="
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background-color: #ff5f56;
                margin-right: 6px;
            "></div>
            <div style="
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background-color: #ffbd2e;
                margin-right: 6px;
            "></div>
            <div style="
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background-color: #27c93f;
                margin-right: 12px;
            "></div>
            {title}
        </div>
        <div style="
            color: #ddd;
            background-color: #1a1a1a;
            padding: 12px;
            font-size: 0.9rem;
            overflow-y: auto;
            height: {height};
            white-space: pre-wrap;
        ">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)

def avatar(image_src: Union[str, Image.Image], size: int = 64, border_color: str = GLIFO_COLOR,
          tooltip: Optional[str] = None):
    """Display a circular avatar image with optional tooltip"""
    if isinstance(image_src, Image.Image):
        # Convert PIL image to base64
        buffered = BytesIO()
        image_src.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        img_src = f"data:image/png;base64,{img_str}"
    else:
        img_src = image_src
    
    tooltip_attr = f'title="{tooltip}"' if tooltip else ''
    
    st.markdown(f"""
    <div {tooltip_attr} style="
        width: {size}px;
        height: {size}px;
        border-radius: 50%;
        overflow: hidden;
        border: 2px solid {border_color};
        display: inline-block;
    ">
        <img src="{img_src}" style="width: 100%; height: 100%; object-fit: cover;">
    </div>
    """, unsafe_allow_html=True)

def animated_gradient_background():
    """Add an animated gradient background to the app"""
    st.markdown("""
    <div class="gradient-background"></div>
    
    <style>
        .gradient-background {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: -1;
            background: linear-gradient(45deg, #0E1117, #1A1D26, #252A37);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }
        
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
    </style>
    """, unsafe_allow_html=True)

def fractal_dashboard():
    """Create a dashboard-specific layout and styling for fractal visualizations"""
    st.markdown("""
    <style>
        /* Fractal dashboard specific styling */
        .fractal-dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        
        .fractal-card {
            background-color: rgba(30, 40, 50, 0.8);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .fractal-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        }
        
        .fractal-header {
            padding: 12px 16px;
            background-color: rgba(110, 68, 255, 0.1);
            border-bottom: 1px solid rgba(110, 68, 255, 0.2);
            font-weight: 500;
        }
        
        .fractal-content {
            padding: 16px;
        }
        
        .fractal-controls {
            background-color: rgba(0,0,0,0.2);
            padding: 12px;
            border-top: 1px solid rgba(255,255,255,0.05);
        }
        
        /* Lemniscate animation */
        .lemniscate-path {
            stroke-dasharray: 1000;
            stroke-dashoffset: 1000;
            animation: dash 10s linear infinite;
        }
        
        @keyframes dash {
            to {
                stroke-dashoffset: 0;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def create_lemniscate_svg(width: int = 200, height: int = 100, color: str = GLIFO_COLOR):
    """Create an SVG lemniscate (infinity symbol) with animation"""
    svg = f"""
    <svg width="{width}" height="{height}" viewBox="0 0 200 100">
        <path class="lemniscate-path" d="M50,50 C50,20 100,20 100,50 C100,80 150,80 150,50 C150,20 100,20 100,50 C100,80 50,80 50,50 Z" 
              fill="none" stroke="{color}" stroke-width="2" />
    </svg>
    """
    return svg

def glifo_pattern_background(opacity: float = 0.2):
    """Add a subtle Glifo pattern background"""
    # Generate a complex pattern with SVG
    pattern = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
        <path d="M25,25 L75,25 L75,75 L25,75 Z" fill="none" stroke="{GLIFO_COLOR}" stroke-width="0.5" opacity="0.3" />
        <path d="M40,40 L60,40 L60,60 L40,60 Z" fill="none" stroke="{GLIFO_COLOR}" stroke-width="0.5" opacity="0.5" />
        <circle cx="50" cy="50" r="15" fill="none" stroke="{GLIFO_COLOR}" stroke-width="0.5" opacity="0.3" />
        <circle cx="50" cy="50" r="25" fill="none" stroke="{GLIFO_COLOR}" stroke-width="0.5" opacity="0.2" />
        <line x1="0" y1="0" x2="100" y2="100" stroke="{GLIFO_COLOR}" stroke-width="0.5" opacity="0.2" />
        <line x1="100" y1="0" x2="0" y2="100" stroke="{GLIFO_COLOR}" stroke-width="0.5" opacity="0.2" />
    </svg>
    """
    
    # Convert SVG to base64
    svg_bytes = pattern.encode('utf-8')
    encoded = base64.b64encode(svg_bytes).decode('utf-8')
    
    st.markdown(f"""
    <style>
        .main .block-container {{
            background-image: url(data:image/svg+xml;base64,{encoded});
            background-repeat: repeat;
            background-opacity: {opacity};
        }}
    </style>
    """, unsafe_allow_html=True)

def apply_wiltonos_theme():
    """Apply all WiltonOS UI enhancements at once"""
    custom_css()
    animated_gradient_background()
    glifo_pattern_background(opacity=0.1)

# Initialize the theme when this module is imported
apply_wiltonos_theme()