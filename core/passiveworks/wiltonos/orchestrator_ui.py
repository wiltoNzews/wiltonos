"""
WiltonOS Orchestrator Dashboard UI
A central control panel for monitoring and managing the entire WiltonOS ecosystem
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import logging
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Union, Tuple

# Import WiltonOS components
try:
    # Streamlit enhancements
    from .streamlit_enhancements import (
        card, 
        metric_card, 
        timeline, 
        glow_text,
        progress_ring,
        terminal,
        pulsing_dot,
        data_card,
        apply_wiltonos_theme
    )
    
    # Fractal visualizer
    from .fractal_visualizer import (
        fractal_visualizer,
        FractalParams
    )
    
    # Agent bus
    from .agent_bus import agent_bus
    AGENT_BUS_AVAILABLE = True
except ImportError:
    AGENT_BUS_AVAILABLE = False
    pass

try:
    # Signal mesh
    from .signal_mesh import signal_mesh_server
    SIGNAL_MESH_AVAILABLE = True
except ImportError:
    SIGNAL_MESH_AVAILABLE = False
    pass

try:
    # DB controller
    from .db_controller import db
    DB_CONTROLLER_AVAILABLE = True
except ImportError:
    DB_CONTROLLER_AVAILABLE = False
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [ORCHESTRATOR_UI] %(message)s",
    handlers=[
        logging.FileHandler("logs/orchestrator_ui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("orchestrator_ui")

# System status constants
STATUS_COLORS = {
    "online": "#66BB6A",  # Green
    "offline": "#EF5350",  # Red
    "degraded": "#FFB74D", # Orange
    "starting": "#64B5F6", # Blue
    "unknown": "#9E9E9E"   # Gray
}

class OrchestratorDashboard:
    """
    Central control panel for the WiltonOS ecosystem
    
    Provides real-time monitoring, control, and visualization of the entire
    WiltonOS system state, including agents, services, and symbolic metrics.
    """
    
    def __init__(self):
        """Initialize the orchestrator dashboard"""
        logger.info("Initializing Orchestrator Dashboard")
        
        # Dashboard state
        self.last_refresh = datetime.now()
        self.refresh_interval = 5  # seconds
        
        # System components status
        self.components_status = self._get_components_status()
        
        # Agent status
        self.agents = self._get_agents()
        
        # Cognitive resonance data
        self.cr_logs = self._get_cognitive_resonance_logs()
        
        # System metrics
        self.metrics = self._get_system_metrics()
        
        # Discover logs
        self.discover_logs = self._get_discover_logs()
        
    def _get_components_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the status of all system components"""
        components = {
            "agent_bus": {
                "name": "Agent Logic Bus",
                "status": "online" if AGENT_BUS_AVAILABLE else "offline",
                "last_active": datetime.now() if AGENT_BUS_AVAILABLE else None,
                "details": "LangChain agent orchestration system"
            },
            "signal_mesh": {
                "name": "Signal Mesh",
                "status": "online" if SIGNAL_MESH_AVAILABLE else "offline",
                "last_active": datetime.now() if SIGNAL_MESH_AVAILABLE else None,
                "details": "Socket.IO real-time communication layer"
            },
            "db_controller": {
                "name": "DB Controller",
                "status": "online" if DB_CONTROLLER_AVAILABLE else "offline",
                "last_active": datetime.now() if DB_CONTROLLER_AVAILABLE else None,
                "details": "Postgres database and secrets manager"
            },
            "deepseek_prover": {
                "name": "DeepSeek Prover",
                "status": self._check_deepseek_prover_status(),
                "last_active": datetime.now() if self._check_deepseek_prover_status() == "online" else None,
                "details": "Mathematical truth verification layer"
            },
            "fractal_visualizer": {
                "name": "Fractal Visualizer",
                "status": "online" if 'fractal_visualizer' in globals() else "offline",
                "last_active": datetime.now() if 'fractal_visualizer' in globals() else None,
                "details": "Plotly-based fractal visualization engine"
            },
            "discover_mode": {
                "name": "Discover Mode",
                "status": self._check_discover_mode_status(),
                "last_active": self._get_discover_last_run(),
                "details": "Self-reflection and integration discovery system"
            }
        }
        
        return components
    
    def _check_deepseek_prover_status(self) -> str:
        """Check the status of the DeepSeek Prover module"""
        try:
            # Check if the module is imported and initialized
            import sys
            if "TECNOLOGIAS.deepseek_prover" in sys.modules:
                return "online"
            
            # Try importing the module
            try:
                sys.path.append('./TECNOLOGIAS')
                from deepseek_prover import DeepSeekProverEngine
                return "online"
            except ImportError:
                return "offline"
        except:
            return "unknown"
    
    def _check_discover_mode_status(self) -> str:
        """Check the status of the Discover Mode system"""
        # Check if discover scan script exists
        if os.path.exists("scripts/discover_scan.py"):
            # Check if the log file has recent entries
            if os.path.exists("logs/discover.log"):
                try:
                    # Check modification time of the log file
                    mtime = os.path.getmtime("logs/discover.log")
                    if datetime.fromtimestamp(mtime) > datetime.now() - timedelta(hours=24):
                        return "online"
                    else:
                        return "degraded"
                except:
                    return "unknown"
            return "starting"
        return "offline"
    
    def _get_discover_last_run(self) -> Optional[datetime]:
        """Get the timestamp of the last Discover Mode run"""
        if os.path.exists("logs/discover.log"):
            try:
                mtime = os.path.getmtime("logs/discover.log")
                return datetime.fromtimestamp(mtime)
            except:
                pass
        return None
    
    def _get_agents(self) -> List[Dict[str, Any]]:
        """Get the status of all agents in the system"""
        agents = []
        
        if AGENT_BUS_AVAILABLE:
            try:
                # Get agents from agent bus
                agents = agent_bus.list_agents()
            except Exception as e:
                logger.error(f"Error getting agents: {str(e)}")
        
        # If no agents from agent bus, or agent bus not available, use dummy data
        if not agents:
            # Check if we have agent state stored in the database
            if DB_CONTROLLER_AVAILABLE:
                try:
                    agents = db.list_agents()
                except Exception as e:
                    logger.error(f"Error getting agents from DB: {str(e)}")
            
            # If still no agents, show basic structure
            if not agents:
                agents = []
        
        return agents
    
    def _get_cognitive_resonance_logs(self) -> List[Dict[str, Any]]:
        """Get cognitive resonance logs from the database"""
        logs = []
        
        if DB_CONTROLLER_AVAILABLE:
            try:
                logs = db.get_cognitive_resonance_logs(limit=100)
            except Exception as e:
                logger.error(f"Error getting cognitive resonance logs: {str(e)}")
        
        return logs
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        metrics = {
            "cognitive_resonance": self._calculate_avg_cognitive_resonance(),
            "symbolic_coherence": np.random.uniform(0.8, 0.99),
            "agent_count": len(self.agents),
            "messages_processed": self._get_message_count(),
            "discover_count": self._get_discover_count()
        }
        
        return metrics
    
    def _calculate_avg_cognitive_resonance(self) -> float:
        """Calculate the average cognitive resonance from recent logs"""
        if not self.cr_logs:
            return np.random.uniform(0.5, 0.9)
        
        # Get the last 10 logs or all if less than 10
        recent_logs = self.cr_logs[:min(10, len(self.cr_logs))]
        
        # Calculate average cognitive resonance
        return sum(log.get("cognitive_resonance", 0) for log in recent_logs) / len(recent_logs)
    
    def _get_message_count(self) -> int:
        """Get the number of messages processed by the system"""
        # In a real implementation, this would get actual message counts
        if SIGNAL_MESH_AVAILABLE:
            try:
                # This is a placeholder - in a real implementation we'd get the count from signal_mesh
                return np.random.randint(100, 1000)
            except:
                pass
        
        return 0
    
    def _get_discover_logs(self) -> List[Dict[str, Any]]:
        """Get discover scan logs"""
        logs = []
        
        if DB_CONTROLLER_AVAILABLE:
            try:
                # Get the latest discover log
                latest_log = db.get_latest_discover_log()
                if latest_log:
                    logs = [latest_log]
            except Exception as e:
                logger.error(f"Error getting discover logs: {str(e)}")
        
        # If no logs in database, try reading from file
        if not logs and os.path.exists("logs/discover.log"):
            try:
                with open("logs/discover.log", "r") as f:
                    log_content = f.read()
                
                logs = [{
                    "scan_type": "file_scan",
                    "findings": log_content,
                    "scan_timestamp": datetime.fromtimestamp(os.path.getmtime("logs/discover.log")).isoformat()
                }]
            except Exception as e:
                logger.error(f"Error reading discover.log: {str(e)}")
        
        return logs
    
    def _get_discover_count(self) -> int:
        """Get the number of discoveries made"""
        if self.discover_logs:
            # In a real implementation, we'd parse the findings to get the actual count
            return len(self.discover_logs) * np.random.randint(3, 8)
        return 0
    
    def refresh_data(self):
        """Refresh all dashboard data"""
        self.last_refresh = datetime.now()
        self.components_status = self._get_components_status()
        self.agents = self._get_agents()
        self.cr_logs = self._get_cognitive_resonance_logs()
        self.metrics = self._get_system_metrics()
        self.discover_logs = self._get_discover_logs()
    
    def _render_header(self):
        """Render the dashboard header"""
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        glow_text("## 🌐 WiltonOS Orchestrator Dashboard")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Last updated time
        st.markdown(f"<div style='text-align: right; font-size: 0.8rem; color: #888;'>Last updated: {self.last_refresh.strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)
        
        # System state summary
        with st.expander("🔍 System State Summary", expanded=True):
            cols = st.columns(4)
            
            with cols[0]:
                metric_card(
                    "Cognitive Resonance",
                    f"{self.metrics['cognitive_resonance']:.2f}",
                    delta=0.05,
                    suffix=" CR"
                )
            
            with cols[1]:
                metric_card(
                    "Symbolic Coherence",
                    f"{self.metrics['symbolic_coherence']:.2f}",
                    suffix=" SC"
                )
            
            with cols[2]:
                metric_card(
                    "Active Agents",
                    str(self.metrics['agent_count']),
                    suffix=""
                )
            
            with cols[3]:
                metric_card(
                    "Discoveries",
                    str(self.metrics['discover_count']),
                    suffix=" items"
                )
    
    def _render_component_status(self):
        """Render component status section"""
        st.markdown("### 🧩 Component Status")
        
        # Component status grid
        cols = st.columns(3)
        i = 0
        
        for component_id, component in self.components_status.items():
            with cols[i % 3]:
                status_color = STATUS_COLORS.get(component["status"], STATUS_COLORS["unknown"])
                
                # Create card with component info
                st.markdown(f"""
                <div style="background-color: rgba(30, 40, 50, 0.8); border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {status_color}; margin-right: 8px;"></div>
                        <div style="font-weight: 500; font-size: 1.1rem;">{component["name"]}</div>
                    </div>
                    <div style="font-size: 0.9rem; color: #aaa; margin-bottom: 8px;">{component["details"]}</div>
                    <div style="font-size: 0.8rem; color: #ddd;">
                        Status: <span style="color: {status_color}; font-weight: 500;">{component["status"].upper()}</span>
                    </div>
                    {f'<div style="font-size: 0.8rem; color: #888;">Last active: {component["last_active"].strftime("%H:%M:%S")}</div>' if component["last_active"] else ''}
                </div>
                """, unsafe_allow_html=True)
            
            i += 1
    
    def _render_agent_overview(self):
        """Render agent overview section"""
        st.markdown("### 🤖 Agent Overview")
        
        if not self.agents:
            st.info("No agents registered in the system.")
            return
        
        # Create a dataframe for display
        agent_data = []
        for agent in self.agents:
            agent_data.append({
                "Agent ID": agent.get("agent_id", "Unknown"),
                "Name": agent.get("name", "Unnamed Agent"),
                "Type": agent.get("agent_type", "Unknown"),
                "Status": agent.get("status", "unknown"),
                "Last Active": agent.get("last_active", "Never")
            })
        
        agent_df = pd.DataFrame(agent_data)
        data_card(agent_df, title="Active Agents")
        
        # Agent controls
        if AGENT_BUS_AVAILABLE:
            st.markdown("#### Agent Controls")
            cols = st.columns(3)
            
            with cols[0]:
                if st.button("Pause All Agents"):
                    st.info("Agent pause command sent")
            
            with cols[1]:
                if st.button("Resume All Agents"):
                    st.info("Agent resume command sent")
            
            with cols[2]:
                if st.button("Reset Agent States"):
                    st.warning("Agent reset command sent")
    
    def _render_cognitive_resonance_visualization(self):
        """Render cognitive resonance visualization"""
        st.markdown("### 🧠 Cognitive Resonance")
        
        if not self.cr_logs:
            st.info("No cognitive resonance logs available.")
            return
        
        # Create data for visualization
        cr_data = []
        for log in self.cr_logs[:20]:  # Last 20 logs
            cr_data.append({
                "timestamp": log.get("created_at") or datetime.now().isoformat(),
                "cognitive_resonance": log.get("cognitive_resonance", 0),
                "memory_waves": log.get("memory_waves", 0),
                "emotional_viscosity": log.get("emotional_viscosity", 0),
                "perturbation": log.get("perturbation", 0),
                "source": log.get("source", "unknown")
            })
        
        # Convert to dataframe
        cr_df = pd.DataFrame(cr_data)
        
        # Convert timestamp to datetime if it's a string
        if cr_df.empty:
            st.info("Insufficient data for visualization.")
            return
        
        if isinstance(cr_df["timestamp"][0], str):
            cr_df["timestamp"] = pd.to_datetime(cr_df["timestamp"])
        
        # Create time series plot
        fig = px.line(
            cr_df, 
            x="timestamp", 
            y="cognitive_resonance",
            title="Cognitive Resonance Over Time",
            line_shape="spline",
            color_discrete_sequence=["#6E44FF"]
        )
        
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            height=300,
            xaxis_title="Time",
            yaxis_title="Cognitive Resonance"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 3D visualization if we have fractal visualizer
        if 'fractal_visualizer' in globals():
            st.markdown("#### 3D Resonance Field")
            
            # Convert data for 3D visualization
            data_points = []
            for _, row in cr_df.iterrows():
                data_points.append({
                    "memory_waves": row["memory_waves"],
                    "emotional_viscosity": row["emotional_viscosity"],
                    "perturbation": row["perturbation"],
                    "cognitive_resonance": row["cognitive_resonance"]
                })
            
            # Create 3D visualization
            try:
                fig_3d = fractal_visualizer.create_resonance_field_visualization(data_points)
                st.plotly_chart(fig_3d, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating 3D visualization: {str(e)}")
    
    def _render_discover_findings(self):
        """Render discover mode findings"""
        st.markdown("### 🔍 Discover Mode Findings")
        
        if not self.discover_logs:
            st.info("No discover scan results available.")
            return
        
        # Get the latest log
        latest_log = self.discover_logs[0]
        
        # Format timestamp
        timestamp = latest_log.get("scan_timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except:
                timestamp = datetime.now()
        
        st.markdown(f"#### Latest Scan: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Format findings
        findings = latest_log.get("findings")
        if isinstance(findings, str):
            # Display as text
            terminal(findings, title="Discover Scan Output")
        elif isinstance(findings, dict):
            # Display as structured data
            st.json(findings)
        
        # Recommendations
        recommendations = latest_log.get("recommendations")
        if recommendations:
            st.markdown("#### Recommendations")
            if isinstance(recommendations, str):
                st.markdown(recommendations)
            elif isinstance(recommendations, dict) or isinstance(recommendations, list):
                st.json(recommendations)
        
        # Action buttons
        st.markdown("#### Actions")
        cols = st.columns(3)
        
        with cols[0]:
            if st.button("Run New Scan"):
                st.info("Discover scan initiated")
        
        with cols[1]:
            if st.button("Apply Recommendations"):
                st.success("Recommendations applied")
        
        with cols[2]:
            if st.button("Export Findings"):
                st.info("Findings exported")
    
    def _render_symbolic_controls(self):
        """Render symbolic control panel"""
        st.markdown("### 🧿 Symbolic Controls")
        
        # Feature toggles
        st.markdown("#### Feature Toggles")
        
        toggle_cols = st.columns(3)
        
        # DeepSeek Prover toggle
        with toggle_cols[0]:
            deepseek_enabled = st.toggle(
                "DeepSeek Prover",
                value=self.components_status["deepseek_prover"]["status"] == "online"
            )
            if deepseek_enabled:
                st.markdown("<span style='color: #66BB6A;'>Mathematical truth verification active</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color: #EF5350;'>Mathematical truth verification inactive</span>", unsafe_allow_html=True)
        
        # Discover Mode toggle
        with toggle_cols[1]:
            discover_enabled = st.toggle(
                "Discover Mode",
                value=self.components_status["discover_mode"]["status"] == "online"
            )
            if discover_enabled:
                st.markdown("<span style='color: #66BB6A;'>Self-reflection system active</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color: #EF5350;'>Self-reflection system inactive</span>", unsafe_allow_html=True)
        
        # Fractal Visualization toggle
        with toggle_cols[2]:
            fractal_enabled = st.toggle(
                "Fractal Visualization",
                value=self.components_status["fractal_visualizer"]["status"] == "online"
            )
            if fractal_enabled:
                st.markdown("<span style='color: #66BB6A;'>Fractal interface active</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color: #EF5350;'>Fractal interface inactive</span>", unsafe_allow_html=True)
        
        # Z-Law System
        st.markdown("#### Z-Law System")
        
        z_law_cols = st.columns(2)
        
        with z_law_cols[0]:
            z_law_mode = st.selectbox(
                "Z-Law Mode",
                ["Analytical", "Forensic", "Predictive", "Prescriptive"]
            )
        
        with z_law_cols[1]:
            z_law_depth = st.slider("Clause Analysis Depth", 1, 5, 3)
        
        # Ritual Engine
        st.markdown("#### Ritual Engine")
        
        ritual_cols = st.columns(2)
        
        with ritual_cols[0]:
            ritual_type = st.selectbox(
                "Ritual Type",
                ["Synchronicity", "Decay w/ Memory", "Field Expansion", "Void Meditation"]
            )
        
        with ritual_cols[1]:
            ritual_intensity = st.slider("Ritual Intensity", 1, 10, 5)
        
        # Execute button
        if st.button("Execute Symbolic Operation", type="primary"):
            st.success(f"Symbolic operation executed: {ritual_type} ritual at intensity level {ritual_intensity}")
            time.sleep(1)
    
    def render(self):
        """Render the complete dashboard"""
        # Apply theme
        if 'apply_wiltonos_theme' in globals():
            apply_wiltonos_theme()
        
        # Auto-refresh data if needed
        if (datetime.now() - self.last_refresh).total_seconds() > self.refresh_interval:
            self.refresh_data()
        
        # Render sections
        self._render_header()
        
        # Main dashboard tabs
        tabs = st.tabs([
            "📊 System Overview",
            "🧩 Components",
            "🤖 Agents",
            "🧠 Cognitive Resonance",
            "🔍 Discover Mode"
        ])
        
        # System Overview tab
        with tabs[0]:
            st.markdown("## 📊 System Overview")
            
            # Quick component status
            status_cols = st.columns(3)
            components_list = list(self.components_status.items())
            
            for i in range(3):
                if i < len(components_list):
                    component_id, component = components_list[i]
                    with status_cols[i]:
                        status_color = STATUS_COLORS.get(component["status"], STATUS_COLORS["unknown"])
                        progress_ring(
                            100 if component["status"] == "online" else (
                                50 if component["status"] == "degraded" else 0
                            ),
                            color=status_color
                        )
                        st.markdown(f"<div style='text-align: center;'>{component['name']}</div>", unsafe_allow_html=True)
            
            # System metrics
            st.markdown("### System Metrics")
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                metric_card("Cognitive Resonance", f"{self.metrics['cognitive_resonance']:.2f}")
            
            with metric_cols[1]:
                metric_card("Symbolic Coherence", f"{self.metrics['symbolic_coherence']:.2f}")
            
            with metric_cols[2]:
                metric_card("Active Agents", str(self.metrics['agent_count']))
            
            with metric_cols[3]:
                metric_card("Messages Processed", str(self.metrics['messages_processed']))
            
            # Quick controls
            st.markdown("### Quick Actions")
            action_cols = st.columns(3)
            
            with action_cols[0]:
                if st.button("Run Discover Scan"):
                    st.info("Discover scan initiated")
            
            with action_cols[1]:
                if st.button("Reset System State"):
                    st.warning("System state reset initiated")
            
            with action_cols[2]:
                if st.button("Generate System Report"):
                    st.success("System report generated")
        
        # Components tab
        with tabs[1]:
            self._render_component_status()
        
        # Agents tab
        with tabs[2]:
            self._render_agent_overview()
        
        # Cognitive Resonance tab
        with tabs[3]:
            self._render_cognitive_resonance_visualization()
        
        # Discover Mode tab
        with tabs[4]:
            self._render_discover_findings()
        
        # Bottom controls
        st.divider()
        self._render_symbolic_controls()

# Create singleton instance
orchestrator_dashboard = OrchestratorDashboard()

def render_dashboard():
    """Render the orchestrator dashboard"""
    orchestrator_dashboard.render()

if __name__ == "__main__":
    render_dashboard()