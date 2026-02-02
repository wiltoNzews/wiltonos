"""
WiltonOS Z-Law Tree Viewer + DeepSeek Integration
Visualize and validate Z-Law clause trees with DeepSeek Prover
"""

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import numpy as np
import json
import logging
import os
import sys
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [ZLAW_TREE] %(message)s",
    handlers=[
        logging.FileHandler("logs/zlaw_tree.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("zlaw_tree")

# Try importing DeepSeek Prover
DEEPSEEK_AVAILABLE = False
try:
    sys.path.append('./TECNOLOGIAS')
    from deepseek_prover import DeepSeekProverEngine
    DEEPSEEK_AVAILABLE = True
    logger.info("DeepSeek Prover successfully imported")
except ImportError:
    logger.warning("DeepSeek Prover not available, verification features will be limited")

# Try importing streamlit enhancements
try:
    from .streamlit_enhancements import (
        card, 
        terminal,
        glow_text,
        code_block,
        apply_wiltonos_theme
    )
except ImportError:
    logger.warning("Streamlit enhancements not available, using standard components")

class ClauseNode:
    """Represents a single clause node in the Z-Law tree"""
    
    def __init__(
        self,
        id: str,
        text: str,
        type: str = "premise",
        certainty: float = 1.0,
        parent_id: Optional[str] = None,
        status: str = "unverified"
    ):
        self.id = id
        self.text = text
        self.type = type  # premise, conclusion, constraint, exception
        self.certainty = certainty  # 0.0 to 1.0
        self.parent_id = parent_id
        self.status = status  # unverified, valid, invalid, uncertain
        self.children = []
        self.verification_result = None
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary"""
        return {
            "id": self.id,
            "text": self.text,
            "type": self.type,
            "certainty": self.certainty,
            "parent_id": self.parent_id,
            "status": self.status,
            "verification_result": self.verification_result,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClauseNode':
        """Create node from dictionary"""
        node = cls(
            id=data["id"],
            text=data["text"],
            type=data.get("type", "premise"),
            certainty=data.get("certainty", 1.0),
            parent_id=data.get("parent_id"),
            status=data.get("status", "unverified")
        )
        node.verification_result = data.get("verification_result")
        node.created_at = data.get("created_at", datetime.now().isoformat())
        return node

class ZLawTree:
    """
    Z-Law Clause Tree Model
    
    Represents a hierarchical structure of logical clauses that can be
    verified for consistency, coherence, and validity using DeepSeek Prover.
    """
    
    def __init__(self, name: str = "New Z-Law Tree"):
        self.name = name
        self.description = ""
        self.root_nodes = []
        self.nodes_by_id = {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.verification_status = "unverified"
        
        # Initialize DeepSeek Prover if available
        self.prover = None
        if DEEPSEEK_AVAILABLE:
            try:
                self.prover = DeepSeekProverEngine()
                logger.info("DeepSeek Prover engine initialized")
            except Exception as e:
                logger.error(f"Error initializing DeepSeek Prover: {str(e)}")
    
    def add_node(
        self,
        text: str,
        type: str = "premise",
        certainty: float = 1.0,
        parent_id: Optional[str] = None
    ) -> ClauseNode:
        """Add a new clause node to the tree"""
        # Generate a unique ID
        node_id = f"node_{len(self.nodes_by_id) + 1}"
        
        # Create the node
        node = ClauseNode(
            id=node_id,
            text=text,
            type=type,
            certainty=certainty,
            parent_id=parent_id,
            status="unverified"
        )
        
        # Add to nodes dictionary
        self.nodes_by_id[node_id] = node
        
        # Add to parent or root nodes
        if parent_id:
            if parent_id in self.nodes_by_id:
                self.nodes_by_id[parent_id].children.append(node)
            else:
                logger.warning(f"Parent node {parent_id} not found, adding as root node")
                self.root_nodes.append(node)
        else:
            self.root_nodes.append(node)
        
        self.updated_at = datetime.now().isoformat()
        logger.info(f"Added node {node_id}: {text[:30]}...")
        
        return node
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its children from the tree"""
        if node_id not in self.nodes_by_id:
            logger.warning(f"Node {node_id} not found")
            return False
        
        node = self.nodes_by_id[node_id]
        
        # Remove from parent's children
        if node.parent_id and node.parent_id in self.nodes_by_id:
            parent = self.nodes_by_id[node.parent_id]
            parent.children = [child for child in parent.children if child.id != node_id]
        
        # Remove from root nodes
        self.root_nodes = [root for root in self.root_nodes if root.id != node_id]
        
        # Remove all children recursively
        children_to_remove = []
        def collect_children(node):
            for child in node.children:
                children_to_remove.append(child.id)
                collect_children(child)
        
        collect_children(node)
        
        # Remove from nodes dictionary
        del self.nodes_by_id[node_id]
        for child_id in children_to_remove:
            if child_id in self.nodes_by_id:
                del self.nodes_by_id[child_id]
        
        self.updated_at = datetime.now().isoformat()
        logger.info(f"Removed node {node_id} and {len(children_to_remove)} children")
        
        return True
    
    def verify_node(self, node_id: str) -> Dict[str, Any]:
        """Verify a node using DeepSeek Prover"""
        if node_id not in self.nodes_by_id:
            logger.warning(f"Node {node_id} not found")
            return {
                "status": "error",
                "message": f"Node {node_id} not found"
            }
        
        node = self.nodes_by_id[node_id]
        
        if not DEEPSEEK_AVAILABLE or not self.prover:
            logger.warning("DeepSeek Prover not available, using simplified verification")
            # Simple syntactic verification
            result = self._simplified_verification(node)
            node.status = result["status"]
            node.verification_result = result
            return result
        
        # Get parent propositions if any
        axioms = []
        if node.parent_id:
            parent_chain = []
            current = node
            while current.parent_id:
                parent = self.nodes_by_id.get(current.parent_id)
                if parent:
                    parent_chain.append(parent)
                    current = parent
                else:
                    break
            
            # Reverse to get ancestors in order
            for parent in reversed(parent_chain):
                axioms.append(parent.text)
        
        # Verify using DeepSeek Prover
        try:
            result = self.prover.verify_proposition(
                proposition=node.text,
                axioms=axioms if axioms else None
            )
            
            logger.info(f"DeepSeek verification for node {node_id}: {result.get('result', 'unknown')}")
            
            # Map DeepSeek result to our status
            status_map = {
                "valid": "valid",
                "proven": "valid",
                "invalid": "invalid",
                "disproven": "invalid",
                "uncertain": "uncertain",
                "unknown": "uncertain"
            }
            
            prover_result = result.get("result", "unknown").lower()
            node.status = status_map.get(prover_result, "uncertain")
            node.verification_result = result
            
            return result
        except Exception as e:
            logger.error(f"Error during DeepSeek verification: {str(e)}")
            result = {
                "status": "error",
                "message": str(e)
            }
            node.status = "error"
            node.verification_result = result
            return result
    
    def _simplified_verification(self, node: ClauseNode) -> Dict[str, Any]:
        """Simplified verification without DeepSeek"""
        # Check for logical structure markers
        text = node.text.lower()
        
        # Look for contradictions
        contradiction_markers = [
            "cannot both be true",
            "contradiction",
            "mutually exclusive",
            "both a and not a",
            "impossible"
        ]
        
        for marker in contradiction_markers:
            if marker in text:
                return {
                    "status": "invalid",
                    "message": f"Possible contradiction detected: '{marker}'",
                    "confidence": 0.7
                }
        
        # Look for valid logical structures
        valid_markers = [
            "therefore",
            "implies",
            "it follows that",
            "consequently",
            "which means"
        ]
        
        for marker in valid_markers:
            if marker in text:
                return {
                    "status": "valid",
                    "message": f"Logical structure detected: '{marker}'",
                    "confidence": 0.8
                }
        
        # Check for uncertainty markers
        uncertainty_markers = [
            "possibly",
            "might be",
            "could be",
            "uncertain",
            "unclear",
            "not proven"
        ]
        
        for marker in uncertainty_markers:
            if marker in text:
                return {
                    "status": "uncertain",
                    "message": f"Uncertainty detected: '{marker}'",
                    "confidence": 0.6
                }
        
        # Default to uncertain with medium confidence
        return {
            "status": "uncertain",
            "message": "No clear logical structure detected",
            "confidence": 0.5
        }
    
    def verify_tree(self) -> Dict[str, Any]:
        """Verify the entire tree"""
        if not self.root_nodes:
            logger.warning("Tree is empty, nothing to verify")
            return {
                "status": "error",
                "message": "Tree is empty"
            }
        
        # Verify each node in breadth-first order
        results = {}
        nodes_to_verify = list(self.root_nodes)
        
        while nodes_to_verify:
            node = nodes_to_verify.pop(0)
            result = self.verify_node(node.id)
            results[node.id] = result
            
            # Add children to the queue
            nodes_to_verify.extend(node.children)
        
        # Determine overall verification status
        statuses = [node.status for node in self.nodes_by_id.values()]
        
        if "invalid" in statuses:
            self.verification_status = "invalid"
        elif "uncertain" in statuses:
            self.verification_status = "uncertain"
        elif "error" in statuses:
            self.verification_status = "error"
        else:
            self.verification_status = "valid"
        
        self.updated_at = datetime.now().isoformat()
        logger.info(f"Verified entire tree: {self.verification_status}")
        
        return {
            "status": self.verification_status,
            "node_results": results,
            "nodes_verified": len(results)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary"""
        # Convert nodes to nested structure
        def build_node_tree(nodes):
            result = []
            for node in nodes:
                node_dict = node.to_dict()
                if node.children:
                    node_dict["children"] = build_node_tree(node.children)
                result.append(node_dict)
            return result
        
        return {
            "name": self.name,
            "description": self.description,
            "nodes": build_node_tree(self.root_nodes),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "verification_status": self.verification_status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ZLawTree':
        """Create tree from dictionary"""
        tree = cls(name=data.get("name", "Loaded Z-Law Tree"))
        tree.description = data.get("description", "")
        tree.created_at = data.get("created_at", datetime.now().isoformat())
        tree.updated_at = data.get("updated_at", datetime.now().isoformat())
        tree.verification_status = data.get("verification_status", "unverified")
        
        # Build node dictionary and reconstruct tree
        def process_nodes(nodes_data, parent_id=None):
            result = []
            for node_data in nodes_data:
                node_data_copy = node_data.copy()
                children_data = node_data_copy.pop("children", [])
                
                node_data_copy["parent_id"] = parent_id
                node = ClauseNode.from_dict(node_data_copy)
                
                tree.nodes_by_id[node.id] = node
                if not parent_id:
                    tree.root_nodes.append(node)
                
                children = process_nodes(children_data, node.id)
                node.children = children
                result.append(node)
            return result
        
        process_nodes(data.get("nodes", []))
        
        return tree
    
    def save_to_file(self, filepath: str) -> bool:
        """Save tree to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Saved tree to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving tree to {filepath}: {str(e)}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str) -> Optional['ZLawTree']:
        """Load tree from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            tree = cls.from_dict(data)
            logger.info(f"Loaded tree from {filepath}")
            return tree
        except Exception as e:
            logger.error(f"Error loading tree from {filepath}: {str(e)}")
            return None

class ZLawTreeVisualizer:
    """
    Z-Law Tree Visualization System
    
    Renders interactive visualizations of Z-Law clause trees,
    showing logical relationships and verification status.
    """
    
    def __init__(self):
        """Initialize the visualizer"""
        pass
    
    def _create_tree_layout(self, tree: ZLawTree) -> Tuple[nx.DiGraph, Dict[str, Tuple[float, float]]]:
        """Create a network layout for the tree"""
        G = nx.DiGraph()
        
        # Add nodes and edges
        for node_id, node in tree.nodes_by_id.items():
            G.add_node(node_id, **node.to_dict())
            if node.parent_id:
                G.add_edge(node.parent_id, node_id)
        
        # Create hierarchical layout
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
        
        return G, pos
    
    def create_tree_visualization(self, tree: ZLawTree) -> go.Figure:
        """Create an interactive visualization of the tree"""
        # Check if tree is empty
        if not tree.root_nodes:
            fig = go.Figure()
            fig.add_annotation(
                text="Empty tree - Add clauses to visualize",
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                template="plotly_dark",
                height=600
            )
            return fig
        
        # Create network layout
        G, pos = self._create_tree_layout(tree)
        
        # Status colors
        status_colors = {
            "unverified": "#9E9E9E",  # Gray
            "valid": "#66BB6A",       # Green
            "invalid": "#EF5350",     # Red
            "uncertain": "#FFB74D",   # Orange
            "error": "#E040FB"        # Purple
        }
        
        # Type symbols
        type_symbols = {
            "premise": "circle",
            "conclusion": "diamond",
            "constraint": "square",
            "exception": "x"
        }
        
        # Create node traces by status
        node_traces = {}
        for status, color in status_colors.items():
            node_traces[status] = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode='markers',
                name=status.capitalize(),
                marker=dict(
                    color=color,
                    size=20,
                    line=dict(width=2, color='#1A1D26')
                ),
                hoverinfo='text'
            )
        
        # Add nodes to appropriate trace
        for node_id, node_data in G.nodes(data=True):
            x, y = pos[node_id]
            status = node_data.get('status', 'unverified')
            if status not in node_traces:
                status = 'unverified'
            
            node_traces[status]['x'] = node_traces[status]['x'] + (x,)
            node_traces[status]['y'] = node_traces[status]['y'] + (y,)
            
            # Hover text
            hover_text = f"ID: {node_id}<br>Type: {node_data.get('type', 'premise')}<br>"
            hover_text += f"Status: {status}<br>Certainty: {node_data.get('certainty', 1.0)}<br>"
            hover_text += f"Text: {node_data.get('text', '')[:50]}..."
            
            node_traces[status]['text'] = node_traces[status]['text'] + (hover_text,)
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=1, color='#757575'),
            mode='lines',
            hoverinfo='none'
        )
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace] + list(node_traces.values()),
            layout=go.Layout(
                title=dict(
                    text=f"Z-Law Tree: {tree.name}",
                    font=dict(size=20)
                ),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template="plotly_dark",
                height=600,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(30,40,50,0.8)"
                )
            )
        )
        
        # Add annotations for node text
        for node_id, node_data in G.nodes(data=True):
            x, y = pos[node_id]
            text = node_data.get('text', '')
            
            # Truncate text if too long
            if len(text) > 30:
                text = text[:30] + "..."
            
            fig.add_annotation(
                x=x,
                y=y - 20,  # Position below the node
                text=text,
                showarrow=False,
                font=dict(
                    size=10,
                    color="#E0E0E0"
                ),
                align="center",
                width=120
            )
        
        return fig
    
    def create_verification_summary(self, tree: ZLawTree) -> go.Figure:
        """Create a summary visualization of verification results"""
        # Count nodes by status
        status_counts = {
            "valid": 0,
            "invalid": 0,
            "uncertain": 0,
            "unverified": 0,
            "error": 0
        }
        
        for node in tree.nodes_by_id.values():
            status = node.status if node.status in status_counts else "unverified"
            status_counts[status] += 1
        
        # Create pie chart
        fig = go.Figure(data=[
            go.Pie(
                labels=list(status_counts.keys()),
                values=list(status_counts.values()),
                hole=0.4,
                marker=dict(
                    colors=[
                        "#66BB6A",  # valid - Green
                        "#EF5350",  # invalid - Red
                        "#FFB74D",  # uncertain - Orange
                        "#9E9E9E",  # unverified - Gray
                        "#E040FB"   # error - Purple
                    ]
                ),
                textinfo='value',
                hoverinfo='label+percent',
                textfont=dict(
                    size=14,
                    color="#FFFFFF"
                )
            )
        ])
        
        fig.update_layout(
            title=dict(
                text=f"Verification Status: {tree.verification_status.capitalize()}",
                font=dict(size=16)
            ),
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig

class ZLawTreeInterface:
    """
    Z-Law Tree Viewer Interface
    
    Streamlit interface for creating, editing, and visualizing Z-Law clause trees
    with DeepSeek Prover verification.
    """
    
    def __init__(self):
        """Initialize the interface"""
        # Initialize session state if needed
        if 'z_law_tree' not in st.session_state:
            st.session_state.z_law_tree = ZLawTree("New Z-Law Tree")
        
        if 'current_node_id' not in st.session_state:
            st.session_state.current_node_id = None
        
        # Initialize the visualizer
        self.visualizer = ZLawTreeVisualizer()
        
        # Search directories for saved trees
        self.saved_trees = self._find_saved_trees()
        
        logger.info("Z-Law Tree Interface initialized")
    
    def _find_saved_trees(self) -> List[str]:
        """Find saved tree files"""
        tree_files = []
        
        # Check in state directory
        if os.path.exists("state"):
            for file in os.listdir("state"):
                if file.endswith(".zlaw.json"):
                    tree_files.append(os.path.join("state", file))
        
        # Check in current directory
        for file in os.listdir("."):
            if file.endswith(".zlaw.json"):
                tree_files.append(file)
        
        return tree_files
    
    def render_tree_editor(self):
        """Render the tree editor interface"""
        st.markdown("### 📝 Z-Law Clause Editor")
        
        # Tree information
        cols = st.columns([3, 1])
        
        with cols[0]:
            tree_name = st.text_input(
                "Tree Name",
                value=st.session_state.z_law_tree.name
            )
            if tree_name != st.session_state.z_law_tree.name:
                st.session_state.z_law_tree.name = tree_name
        
        with cols[1]:
            verification_status = st.session_state.z_law_tree.verification_status
            status_color = {
                "unverified": "#9E9E9E",
                "valid": "#66BB6A",
                "invalid": "#EF5350", 
                "uncertain": "#FFB74D",
                "error": "#E040FB"
            }.get(verification_status, "#9E9E9E")
            
            st.markdown(f"""
            <div style="background-color: rgba(30, 40, 50, 0.8); padding: 10px; border-radius: 5px; text-align: center;">
                <div style="font-size: 0.8rem; color: #888;">Verification Status</div>
                <div style="color: {status_color}; font-weight: bold; font-size: 1.1rem;">
                    {verification_status.upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        tree_description = st.text_area(
            "Tree Description",
            value=st.session_state.z_law_tree.description
        )
        if tree_description != st.session_state.z_law_tree.description:
            st.session_state.z_law_tree.description = tree_description
        
        # Node editor
        st.markdown("#### Add New Clause")
        
        # Parent selector
        parent_options = [("None", None)] + [(f"{node.id}: {node.text[:30]}...", node.id) 
                                          for node in st.session_state.z_law_tree.nodes_by_id.values()]
        
        parent_display, parent_id = parent_options[0]
        if len(parent_options) > 1:
            parent_display = st.selectbox(
                "Parent Clause",
                options=[p[0] for p in parent_options],
                index=0
            )
            parent_id = next(p[1] for p in parent_options if p[0] == parent_display)
        
        # Clause type
        clause_type = st.selectbox(
            "Clause Type",
            options=["premise", "conclusion", "constraint", "exception"],
            index=0
        )
        
        # Certainty slider
        certainty = st.slider(
            "Certainty",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1
        )
        
        # Clause text
        clause_text = st.text_area(
            "Clause Text",
            value="",
            height=100,
            placeholder="Enter logical clause text here..."
        )
        
        # Add button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Add Clause", type="primary"):
                if clause_text.strip():
                    node = st.session_state.z_law_tree.add_node(
                        text=clause_text,
                        type=clause_type,
                        certainty=certainty,
                        parent_id=parent_id
                    )
                    st.success(f"Added clause: {node.id}")
                    st.session_state.current_node_id = node.id
                else:
                    st.error("Clause text cannot be empty")
    
    def render_tree_viewer(self):
        """Render the tree visualization"""
        st.markdown("### 🌳 Z-Law Tree Visualization")
        
        # Create tree visualization
        fig = self.visualizer.create_tree_visualization(st.session_state.z_law_tree)
        st.plotly_chart(fig, use_container_width=True)
        
        # Verification summary if tree has nodes
        if st.session_state.z_law_tree.nodes_by_id:
            summary_fig = self.visualizer.create_verification_summary(st.session_state.z_law_tree)
            st.plotly_chart(summary_fig, use_container_width=True)
    
    def render_node_inspector(self):
        """Render the node inspector panel"""
        st.markdown("### 🔍 Clause Inspector")
        
        # Check if tree has nodes
        if not st.session_state.z_law_tree.nodes_by_id:
            st.info("No clauses to inspect. Add clauses using the editor above.")
            return
        
        # Node selector
        node_options = [(f"{node.id}: {node.text[:30]}...", node.id) 
                       for node in st.session_state.z_law_tree.nodes_by_id.values()]
        
        node_display, node_id = node_options[0]
        
        # Set selected node from session state if exists
        if st.session_state.current_node_id in st.session_state.z_law_tree.nodes_by_id:
            current_display = next((display for display, nid in node_options 
                                 if nid == st.session_state.current_node_id), None)
            if current_display:
                node_display = current_display
                node_id = st.session_state.current_node_id
        
        # Create selectbox
        selected_display = st.selectbox(
            "Select Clause",
            options=[n[0] for n in node_options],
            index=next((i for i, n in enumerate(node_options) if n[0] == node_display), 0)
        )
        
        # Get node ID from display
        selected_id = next(n[1] for n in node_options if n[0] == selected_display)
        st.session_state.current_node_id = selected_id
        
        # Get the node
        node = st.session_state.z_law_tree.nodes_by_id[selected_id]
        
        # Show node details
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"#### {node.type.capitalize()}: {node.id}")
            
            # Node text with code formatting
            if 'code_block' in globals():
                code_block(node.text)
            else:
                st.code(node.text)
        
        with col2:
            # Status indicator
            status_color = {
                "unverified": "#9E9E9E",
                "valid": "#66BB6A",
                "invalid": "#EF5350", 
                "uncertain": "#FFB74D",
                "error": "#E040FB"
            }.get(node.status, "#9E9E9E")
            
            st.markdown(f"""
            <div style="background-color: rgba(30, 40, 50, 0.8); padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 10px;">
                <div style="font-size: 0.8rem; color: #888;">Status</div>
                <div style="color: {status_color}; font-weight: bold; font-size: 1.1rem;">
                    {node.status.upper()}
                </div>
            </div>
            
            <div style="background-color: rgba(30, 40, 50, 0.8); padding: 10px; border-radius: 5px; text-align: center;">
                <div style="font-size: 0.8rem; color: #888;">Certainty</div>
                <div style="font-weight: bold; font-size: 1.1rem;">
                    {node.certainty:.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Verification details
        st.markdown("#### Verification Details")
        
        # Check if node has been verified
        if node.verification_result:
            if isinstance(node.verification_result, dict):
                # Format verification result
                if 'terminal' in globals():
                    terminal(json.dumps(node.verification_result, indent=2), 
                          title="Verification Result")
                else:
                    st.json(node.verification_result)
            else:
                st.markdown(str(node.verification_result))
        else:
            st.info("Clause has not been verified yet.")
        
        # Verification buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Verify Clause", key="verify_node"):
                with st.spinner("Verifying clause..."):
                    result = st.session_state.z_law_tree.verify_node(node.id)
                    if result:
                        st.success(f"Clause verified: {result.get('status', 'unknown')}")
                    else:
                        st.error("Verification failed")
                    
                    # Force refresh
                    st.experimental_rerun()
        
        with col2:
            if st.button("Delete Clause", key="delete_node"):
                if st.session_state.z_law_tree.remove_node(node.id):
                    st.success(f"Clause {node.id} deleted")
                    st.session_state.current_node_id = None
                    st.experimental_rerun()
                else:
                    st.error(f"Failed to delete clause {node.id}")
    
    def render_tree_controls(self):
        """Render tree control panel"""
        st.markdown("### 🛠️ Tree Controls")
        
        col1, col2, col3 = st.columns(3)
        
        # Verify entire tree
        with col1:
            if st.button("Verify Entire Tree", key="verify_tree"):
                if not st.session_state.z_law_tree.nodes_by_id:
                    st.warning("Tree is empty, nothing to verify")
                else:
                    with st.spinner("Verifying tree..."):
                        result = st.session_state.z_law_tree.verify_tree()
                        st.success(f"Tree verified: {result.get('status', 'unknown')}")
                        st.experimental_rerun()
        
        # Save tree
        with col2:
            if st.button("Save Tree", key="save_tree"):
                # Default name based on tree name
                file_name = st.session_state.z_law_tree.name.lower().replace(" ", "_") + ".zlaw.json"
                file_path = os.path.join("state", file_name)
                
                # Create state directory if it doesn't exist
                os.makedirs("state", exist_ok=True)
                
                if st.session_state.z_law_tree.save_to_file(file_path):
                    st.success(f"Tree saved to {file_path}")
                    self.saved_trees = self._find_saved_trees()
                else:
                    st.error(f"Failed to save tree to {file_path}")
        
        # New tree
        with col3:
            if st.button("New Tree", key="new_tree"):
                st.session_state.z_law_tree = ZLawTree("New Z-Law Tree")
                st.session_state.current_node_id = None
                st.success("Created new tree")
                st.experimental_rerun()
        
        # Load tree
        if self.saved_trees:
            st.markdown("#### Load Saved Tree")
            selected_tree = st.selectbox(
                "Select Tree to Load",
                options=self.saved_trees
            )
            
            if st.button("Load Selected Tree"):
                loaded_tree = ZLawTree.load_from_file(selected_tree)
                if loaded_tree:
                    st.session_state.z_law_tree = loaded_tree
                    st.session_state.current_node_id = None
                    st.success(f"Loaded tree: {loaded_tree.name}")
                    st.experimental_rerun()
                else:
                    st.error(f"Failed to load tree from {selected_tree}")
    
    def render_deepseek_info(self):
        """Render DeepSeek Prover information panel"""
        st.markdown("### 🧮 DeepSeek Prover Integration")
        
        if DEEPSEEK_AVAILABLE and st.session_state.z_law_tree.prover:
            st.markdown("""
            <div style="background-color: rgba(30, 40, 50, 0.8); padding: 15px; border-radius: 10px; border-left: 4px solid #66BB6A;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #66BB6A; margin-right: 8px;"></div>
                    <div style="font-weight: 500; font-size: 1.1rem;">DeepSeek Prover Connected</div>
                </div>
                <div style="margin-top: 10px; font-size: 0.9rem;">
                    DeepSeek Prover V2-671B is providing full mathematical and logical verification capabilities.
                    Clauses will be verified for logical consistency, coherence, and validity.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: rgba(30, 40, 50, 0.8); padding: 15px; border-radius: 10px; border-left: 4px solid #EF5350;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #EF5350; margin-right: 8px;"></div>
                    <div style="font-weight: 500; font-size: 1.1rem;">DeepSeek Prover Not Available</div>
                </div>
                <div style="margin-top: 10px; font-size: 0.9rem;">
                    Using simplified verification based on syntactic analysis. For full logical verification capabilities,
                    ensure DeepSeek Prover module is available in the TECNOLOGIAS directory.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show import instructions
            with st.expander("How to Enable DeepSeek Prover"):
                st.markdown("""
                1. Ensure the DeepSeek Prover module is available in the `TECNOLOGIAS` directory
                2. The module should be named `deepseek_prover.py`
                3. The module should contain the `DeepSeekProverEngine` class
                4. Restart the application after adding the module
                """)
    
    def render(self):
        """Render the main interface"""
        # Apply WiltonOS theme if available
        if 'apply_wiltonos_theme' in globals():
            apply_wiltonos_theme()
        
        # Title and introduction
        if 'glow_text' in globals():
            glow_text("# Z-Law Tree Viewer + DeepSeek Integration")
        else:
            st.title("Z-Law Tree Viewer + DeepSeek Integration")
        
        st.markdown("""
        Visualize, analyze, and verify logical clause trees using DeepSeek Prover.
        Create complex logical structures and verify their consistency, coherence, and validity.
        """)
        
        # Main tabs
        tabs = st.tabs([
            "📊 Tree Visualization", 
            "📝 Clause Editor",
            "🔍 Clause Inspector", 
            "🛠️ Controls"
        ])
        
        # Tree Visualization tab
        with tabs[0]:
            self.render_tree_viewer()
        
        # Clause Editor tab
        with tabs[1]:
            self.render_tree_editor()
        
        # Clause Inspector tab
        with tabs[2]:
            self.render_node_inspector()
        
        # Controls tab
        with tabs[3]:
            self.render_tree_controls()
            st.divider()
            self.render_deepseek_info()

# Create singleton instance
zlaw_tree_interface = ZLawTreeInterface()

def render_interface():
    """Render the Z-Law Tree interface"""
    zlaw_tree_interface.render()

if __name__ == "__main__":
    render_interface()