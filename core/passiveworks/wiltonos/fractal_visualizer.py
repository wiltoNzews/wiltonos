"""
WiltonOS Fractal Visualizer
Interactive fractal pattern visualization using Plotly
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
import logging
import random
import json
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [FRACTAL_VIZ] %(message)s",
    handlers=[
        logging.FileHandler("logs/fractal_visualizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fractal_visualizer")

@dataclass
class FractalParams:
    """Parameters for fractal visualization"""
    type: str = "mandelbrot"
    iterations: int = 100
    x_range: Tuple[float, float] = (-2.0, 1.0)
    y_range: Tuple[float, float] = (-1.5, 1.5)
    resolution: int = 500
    color_scale: str = "Viridis"
    z_formula: str = "log(z + 1)"
    julia_constant: Optional[complex] = None
    decay_rate: float = 0.02
    lemniscate_factor: float = 0.5

class FractalVisualizer:
    """
    Interactive Fractal Visualization System for WiltonOS
    
    Generates interactive visualizations of various fractal systems including:
    - Mandelbrot Set
    - Julia Sets
    - Burning Ship Fractals
    - Lemniscate Patterns
    - Fractal Decay Patterns
    """
    
    def __init__(self):
        logger.info("Fractal Visualizer initialized")
    
    def _mandelbrot(self, h: int, w: int, max_iters: int, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> np.ndarray:
        """Compute the Mandelbrot fractal"""
        y, x = np.ogrid[y_range[0]:y_range[1]:h*1j, x_range[0]:x_range[1]:w*1j]
        c = x + y*1j
        z = c
        divtime = max_iters + np.zeros(z.shape, dtype=int)
        
        for i in range(max_iters):
            z = z**2 + c
            diverge = z*np.conj(z) > 2**2
            div_now = diverge & (divtime == max_iters)
            divtime[div_now] = i
            z[diverge] = 2
        
        return divtime
    
    def _julia(self, h: int, w: int, max_iters: int, 
              x_range: Tuple[float, float], y_range: Tuple[float, float], 
              c: complex) -> np.ndarray:
        """Compute the Julia fractal for a given constant c"""
        y, x = np.ogrid[y_range[0]:y_range[1]:h*1j, x_range[0]:x_range[1]:w*1j]
        z = x + y*1j
        divtime = max_iters + np.zeros(z.shape, dtype=int)
        
        for i in range(max_iters):
            z = z**2 + c
            diverge = z*np.conj(z) > 2**2
            div_now = diverge & (divtime == max_iters)
            divtime[div_now] = i
            z[diverge] = 2
        
        return divtime
    
    def _burning_ship(self, h: int, w: int, max_iters: int, 
                     x_range: Tuple[float, float], y_range: Tuple[float, float]) -> np.ndarray:
        """Compute the Burning Ship fractal"""
        y, x = np.ogrid[y_range[0]:y_range[1]:h*1j, x_range[0]:x_range[1]:w*1j]
        c = x + y*1j
        z = c
        divtime = max_iters + np.zeros(z.shape, dtype=int)
        
        for i in range(max_iters):
            z = (abs(z.real) + abs(z.imag)*1j)**2 + c
            diverge = z*np.conj(z) > 2**2
            div_now = diverge & (divtime == max_iters)
            divtime[div_now] = i
            z[diverge] = 2
        
        return divtime
    
    def _apply_z_transform(self, z: np.ndarray, formula: str) -> np.ndarray:
        """Apply a transformation to the z values"""
        # Define a dictionary of available formulas
        formulas = {
            "log(z + 1)": lambda z: np.log(z + 1),
            "sqrt(z)": lambda z: np.sqrt(z),
            "z": lambda z: z,
            "sin(z)": lambda z: np.sin(z),
            "z^2": lambda z: z**2,
            "1/z": lambda z: 1/np.maximum(z, 0.001),  # Avoid division by zero
            "exp(-z)": lambda z: np.exp(-z)
        }
        
        if formula in formulas:
            return formulas[formula](z)
        else:
            logger.warning(f"Unknown z-transform: {formula}, using default")
            return np.log(z + 1)
    
    def _create_lemniscate_pattern(self, h: int, w: int, factor: float = 0.5) -> np.ndarray:
        """Create a lemniscate pattern based on Bernoulli's lemniscate formula"""
        x = np.linspace(-2, 2, w)
        y = np.linspace(-2, 2, h)
        X, Y = np.meshgrid(x, y)
        
        # Lemniscate formula: (x^2 + y^2)^2 = a^2 * (x^2 - y^2)
        a = factor * 2
        Z = np.abs((X**2 + Y**2)**2 - a**2 * (X**2 - Y**2))
        Z = 1 / (1 + Z)  # Normalize and invert so the lemniscate is higher
        
        return Z
    
    def _create_decay_pattern(self, fractal: np.ndarray, decay_rate: float = 0.02) -> np.ndarray:
        """Apply a decay pattern to a fractal"""
        h, w = fractal.shape
        y, x = np.ogrid[0:h, 0:w]
        
        # Calculate distance from center
        center_y, center_x = h//2, w//2
        distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        
        # Normalize distance
        max_distance = np.sqrt(center_y**2 + center_x**2)
        normalized_distance = distance / max_distance
        
        # Apply exponential decay
        decay = np.exp(-decay_rate * normalized_distance * 10)
        
        # Apply decay to fractal
        decayed_fractal = fractal * decay
        
        return decayed_fractal
    
    def create_fractal_figure(self, params: FractalParams) -> go.Figure:
        """Create an interactive Plotly figure of a fractal"""
        logger.info(f"Creating fractal: {params.type}")
        
        h, w = params.resolution, params.resolution
        
        # Generate the fractal data
        if params.type == "mandelbrot":
            fractal = self._mandelbrot(h, w, params.iterations, params.x_range, params.y_range)
        elif params.type == "julia":
            c = params.julia_constant or complex(-0.8, 0.156)
            fractal = self._julia(h, w, params.iterations, params.x_range, params.y_range, c)
        elif params.type == "burning_ship":
            fractal = self._burning_ship(h, w, params.iterations, params.x_range, params.y_range)
        elif params.type == "lemniscate":
            fractal = self._create_lemniscate_pattern(h, w, params.lemniscate_factor)
        else:
            logger.warning(f"Unknown fractal type: {params.type}, defaulting to Mandelbrot")
            fractal = self._mandelbrot(h, w, params.iterations, params.x_range, params.y_range)
        
        # Apply decay if needed
        if params.decay_rate > 0:
            fractal = self._create_decay_pattern(fractal, params.decay_rate)
        
        # Apply z-transform
        z_values = self._apply_z_transform(fractal, params.z_formula)
        
        # Create the figure
        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            colorscale=params.color_scale,
            zsmooth='best'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{params.type.capitalize()} Fractal",
            xaxis=dict(
                title="Re(c)" if params.type != "lemniscate" else "x",
                showgrid=False,
                zeroline=False,
                range=params.x_range
            ),
            yaxis=dict(
                title="Im(c)" if params.type != "lemniscate" else "y",
                showgrid=False,
                zeroline=False,
                range=params.y_range,
                scaleanchor="x",
                scaleratio=1
            ),
            width=800,
            height=800,
            template="plotly_dark"
        )
        
        return fig
    
    def create_lemniscate_animation(self, frames: int = 60, factor_range: Tuple[float, float] = (0.1, 1.0), 
                                   resolution: int = 300) -> go.Figure:
        """Create an animated lemniscate figure"""
        logger.info("Creating lemniscate animation")
        
        # Create frames
        frame_data = []
        factors = np.linspace(factor_range[0], factor_range[1], frames//2)
        factors = np.concatenate([factors, factors[::-1]])  # Full cycle
        
        for factor in factors:
            params = FractalParams(
                type="lemniscate",
                resolution=resolution,
                lemniscate_factor=factor
            )
            lemniscate = self._create_lemniscate_pattern(resolution, resolution, factor)
            frame_data.append(lemniscate)
        
        # Create figure with animation
        fig = go.Figure(
            data=[go.Heatmap(z=frame_data[0], colorscale="Viridis", zsmooth='best')],
            layout=go.Layout(
                title="Lemniscate Animation",
                xaxis=dict(showgrid=False, zeroline=False, range=[-2, 2]),
                yaxis=dict(showgrid=False, zeroline=False, range=[-2, 2], scaleanchor="x", scaleratio=1),
                width=800,
                height=800,
                template="plotly_dark",
                updatemenus=[{
                    "type": "buttons",
                    "buttons": [
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                        )
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 10},
                    "showactive": False,
                    "x": 0.1,
                    "y": 0,
                    "xanchor": "right",
                    "yanchor": "top"
                }]
            ),
            frames=[go.Frame(data=[go.Heatmap(z=frame)]) for frame in frame_data]
        )
        
        return fig
    
    def create_fractal_decay_figure(self, base_type: str = "mandelbrot", stages: int = 5) -> go.Figure:
        """Create a multi-panel figure showing fractal decay stages"""
        logger.info(f"Creating fractal decay visualization with {stages} stages")
        
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=stages,
            subplot_titles=[f"Stage {i+1}" for i in range(stages)]
        )
        
        # Base parameters
        base_params = FractalParams(
            type=base_type,
            resolution=300,
            iterations=100
        )
        
        # Generate fractals with increasing decay
        for i in range(stages):
            decay_rate = i * 0.01 + 0.005  # Increasing decay
            params = FractalParams(
                type=base_params.type,
                resolution=base_params.resolution,
                iterations=base_params.iterations,
                x_range=base_params.x_range,
                y_range=base_params.y_range,
                decay_rate=decay_rate
            )
            
            # Generate fractal
            if params.type == "mandelbrot":
                fractal = self._mandelbrot(params.resolution, params.resolution, 
                                          params.iterations, params.x_range, params.y_range)
            elif params.type == "julia":
                c = params.julia_constant or complex(-0.8, 0.156)
                fractal = self._julia(params.resolution, params.resolution, 
                                     params.iterations, params.x_range, params.y_range, c)
            else:
                fractal = self._mandelbrot(params.resolution, params.resolution, 
                                          params.iterations, params.x_range, params.y_range)
            
            # Apply decay
            decayed = self._create_decay_pattern(fractal, params.decay_rate)
            
            # Apply z-transform
            z_values = self._apply_z_transform(decayed, params.z_formula)
            
            # Add to subplot
            fig.add_trace(
                go.Heatmap(
                    z=z_values,
                    colorscale=params.color_scale,
                    zsmooth='best',
                    showscale=i == stages-1  # Only show colorbar for last subplot
                ),
                row=1, col=i+1
            )
        
        # Update layout
        fig.update_layout(
            title_text=f"{base_type.capitalize()} Fractal Decay Process",
            height=500,
            width=1200,
            template="plotly_dark"
        )
        
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        
        return fig
    
    def create_resonance_field_visualization(self, data_points: List[Dict[str, Any]]) -> go.Figure:
        """
        Create a 3D visualization of a cognitive resonance field from data points
        
        Args:
            data_points: List of dictionaries with memory_waves, emotional_viscosity,
                        perturbation, and cognitive_resonance values
        """
        logger.info(f"Creating resonance field visualization with {len(data_points)} points")
        
        # Extract data
        x = [point.get('memory_waves', 0) for point in data_points]
        y = [point.get('emotional_viscosity', 0) for point in data_points]
        z = [point.get('perturbation', 0) for point in data_points]
        colors = [point.get('cognitive_resonance', 0) for point in data_points]
        
        # Create figure
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Cognitive Resonance")
            ),
            text=[f"CR: {cr:.2f}" for cr in colors],
            hoverinfo="text"
        )])
        
        # Add lemniscate path if enough points
        if len(data_points) >= 8:
            # Create a lemniscate-like path through the points
            t = np.linspace(0, 2*np.pi, 100)
            a = 0.5  # Shape parameter
            lemniscate_x = a * np.sqrt(2) * np.cos(t) / (np.sin(t)**2 + 1)
            lemniscate_y = a * np.sqrt(2) * np.cos(t) * np.sin(t) / (np.sin(t)**2 + 1)
            
            # Scale and shift to match data range
            x_range = max(x) - min(x)
            y_range = max(y) - min(y)
            x_mid = (max(x) + min(x)) / 2
            y_mid = (max(y) + min(y)) / 2
            
            scale = min(x_range, y_range) / 2
            lemniscate_x = lemniscate_x * scale + x_mid
            lemniscate_y = lemniscate_y * scale + y_mid
            
            # Generate z-values based on x,y position
            lemniscate_z = np.zeros_like(lemniscate_x)
            for i in range(len(lemniscate_x)):
                # Find nearest data point
                distances = [(lemniscate_x[i] - x[j])**2 + (lemniscate_y[i] - y[j])**2 for j in range(len(x))]
                nearest = np.argmin(distances)
                lemniscate_z[i] = z[nearest]
            
            # Add the lemniscate curve
            fig.add_trace(go.Scatter3d(
                x=lemniscate_x,
                y=lemniscate_y,
                z=lemniscate_z,
                mode='lines',
                line=dict(
                    color='rgba(255, 255, 255, 0.8)',
                    width=4
                ),
                name="Lemniscate Path"
            ))
        
        # Update layout
        fig.update_layout(
            title="Cognitive Resonance Field",
            scene=dict(
                xaxis_title="Memory Waves",
                yaxis_title="Emotional Viscosity",
                zaxis_title="Perturbation",
                xaxis=dict(range=[0, 10]),
                yaxis=dict(range=[0, 10]),
                zaxis=dict(range=[0, 10])
            ),
            width=800,
            height=800,
            template="plotly_dark"
        )
        
        return fig
    
    def save_figure_to_json(self, fig: go.Figure, filepath: str) -> bool:
        """Save a Plotly figure to a JSON file"""
        try:
            fig_json = fig.to_json()
            with open(filepath, 'w') as f:
                f.write(fig_json)
            logger.info(f"Figure saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving figure: {str(e)}")
            return False
    
    def load_figure_from_json(self, filepath: str) -> Optional[go.Figure]:
        """Load a Plotly figure from a JSON file"""
        try:
            with open(filepath, 'r') as f:
                fig_json = f.read()
            fig = go.Figure(json.loads(fig_json))
            logger.info(f"Figure loaded from {filepath}")
            return fig
        except Exception as e:
            logger.error(f"Error loading figure: {str(e)}")
            return None

# Create a singleton instance
fractal_visualizer = FractalVisualizer()