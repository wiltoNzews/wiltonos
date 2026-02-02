"""
WiltonOS Streamlit Integration
Integrates the new WiltonOS components into the main Streamlit application
"""

import streamlit as st
import logging
import sys
import os
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [ST_INTEGRATION] %(message)s",
    handlers=[
        logging.FileHandler("logs/streamlit_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("streamlit_integration")

# Import WiltonOS components
ORCHESTRATOR_AVAILABLE = False
ZLAW_TREE_AVAILABLE = False
RITUAL_ENGINE_AVAILABLE = False

try:
    from .orchestrator_ui import render_dashboard
    ORCHESTRATOR_AVAILABLE = True
    logger.info("Orchestrator UI loaded successfully")
except ImportError as e:
    logger.warning(f"Orchestrator UI not available: {str(e)}")
    
    def render_dashboard():
        st.warning("Orchestrator UI not available. Make sure the orchestrator_ui.py module is properly installed.")

try:
    from .zlaw_tree import render_interface as render_zlaw_interface
    ZLAW_TREE_AVAILABLE = True
    logger.info("Z-Law Tree Viewer loaded successfully")
except ImportError as e:
    logger.warning(f"Z-Law Tree Viewer not available: {str(e)}")
    
    def render_zlaw_interface():
        st.warning("Z-Law Tree Viewer not available. Make sure the zlaw_tree.py module is properly installed.")

try:
    from .ritual_engine import render_interface as render_ritual_interface
    RITUAL_ENGINE_AVAILABLE = True
    logger.info("Ritual Engine loaded successfully")
except ImportError as e:
    logger.warning(f"Ritual Engine not available: {str(e)}")
    
    def render_ritual_interface():
        st.warning("Ritual Engine not available. Make sure the ritual_engine.py module is properly installed.")

class StreamlitIntegration:
    """
    WiltonOS Streamlit Integration
    
    Handles the integration of WiltonOS components into the main Streamlit app,
    providing a unified interface and navigation.
    """
    
    def __init__(self):
        """Initialize the integration"""
        logger.info("Initializing WiltonOS Streamlit Integration")
        
        # Register components
        self.components = {
            "orchestrator": {
                "name": "WiltonOS Orchestrator",
                "render_function": render_dashboard,
                "available": ORCHESTRATOR_AVAILABLE,
                "icon": "🌐",
                "description": "Central control panel for monitoring and managing the entire WiltonOS ecosystem"
            },
            "zlaw_tree": {
                "name": "Z-Law Tree Viewer",
                "render_function": render_zlaw_interface,
                "available": ZLAW_TREE_AVAILABLE,
                "icon": "🧠",
                "description": "Visualize and validate Z-Law clause trees with DeepSeek Prover"
            },
            "ritual_engine": {
                "name": "Ritual Engine",
                "render_function": render_ritual_interface,
                "available": RITUAL_ENGINE_AVAILABLE,
                "icon": "🪄",
                "description": "Define, perform, and track repeatable symbolic actions with time and location awareness"
            }
        }
        
        # Cache component availability
        self.available_components = {k: v for k, v in self.components.items() if v["available"]}
        logger.info(f"Available components: {list(self.available_components.keys())}")
    
    def render_component(self, component_id: str):
        """Render a specific component"""
        if component_id not in self.components:
            st.error(f"Component {component_id} not found")
            return
        
        component = self.components[component_id]
        
        if not component["available"]:
            st.warning(f"{component['name']} is not available")
            return
        
        # Render the component
        component["render_function"]()
    
    def render_component_selector(self):
        """Render a component selector interface"""
        st.markdown("## WiltonOS Components")
        
        # Create component cards
        cols = st.columns(len(self.available_components))
        
        for i, (component_id, component) in enumerate(self.available_components.items()):
            with cols[i]:
                st.markdown(f"""
                <div style="background-color: rgba(30, 40, 50, 0.8); border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #6E44FF; height: 150px;">
                    <div style="font-size: 2rem; margin-bottom: 10px;">{component['icon']}</div>
                    <div style="font-weight: 500; font-size: 1.1rem; margin-bottom: 5px;">{component['name']}</div>
                    <div style="font-size: 0.9rem; color: #ddd;">{component['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Open {component['name']}", key=f"open_{component_id}"):
                    st.session_state.current_component = component_id
                    st.experimental_rerun()
    
    def render(self):
        """Render the integration UI"""
        # Initialize session state for component navigation
        if "current_component" not in st.session_state:
            st.session_state.current_component = None
        
        # Show component selector if no component is selected
        if not st.session_state.current_component:
            self.render_component_selector()
            return
        
        # Render the selected component
        component_id = st.session_state.current_component
        
        # Add a back button
        if st.button("← Back to Components"):
            st.session_state.current_component = None
            st.experimental_rerun()
        
        # Render the component
        self.render_component(component_id)

# Create singleton instance
streamlit_integration = StreamlitIntegration()

def render_integration():
    """Render the streamlit integration"""
    streamlit_integration.render()

def register_in_main_app(app_module):
    """
    Register the WiltonOS components in the main Streamlit app
    
    Args:
        app_module: The main app module
    """
    # Check if app module has tabs attribute
    if not hasattr(app_module, "tabs"):
        logger.warning("Main app module does not have tabs attribute")
        return False
    
    try:
        # Add tabs for each component
        for component_id, component in streamlit_integration.available_components.items():
            # Check if the component is already registered
            tab_names = [tab.label for tab in app_module.tabs]
            if f"{component['icon']} {component['name']}" in tab_names:
                logger.info(f"Component {component_id} already registered")
                continue
            
            # Add the component tab
            app_module.tabs.append({
                "label": f"{component['icon']} {component['name']}",
                "render_function": lambda comp_id=component_id: streamlit_integration.render_component(comp_id)
            })
            
            logger.info(f"Component {component_id} registered in main app")
        
        return True
    except Exception as e:
        logger.error(f"Error registering components in main app: {str(e)}")
        return False

# Entry point for direct execution
if __name__ == "__main__":
    render_integration()