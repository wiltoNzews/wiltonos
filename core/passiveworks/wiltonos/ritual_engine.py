"""
WiltonOS Ritual Engine
Define, execute, and track repeatable symbolic actions with time and location awareness
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import logging
import os
import sys
import re
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pytz
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [RITUAL_ENGINE] %(message)s",
    handlers=[
        logging.FileHandler("logs/ritual_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ritual_engine")

# Try importing Whisper integration
WHISPER_AVAILABLE = False
try:
    # This is a placeholder - in your actual implementation, 
    # this would import the Whisper components
    if os.path.exists("wiltonos/whisper_flow.py"):
        from .whisper_flow import whisper_transcribe
        WHISPER_AVAILABLE = True
        logger.info("Whisper integration available")
except ImportError:
    logger.warning("Whisper integration not available")

# Try importing streamlit enhancements
try:
    from .streamlit_enhancements import (
        card, 
        timeline,
        glow_text,
        progress_ring,
        pulsing_dot,
        apply_wiltonos_theme
    )
except ImportError:
    logger.warning("Streamlit enhancements not available, using standard components")

# Try importing fractal visualizer
try:
    from .fractal_visualizer import (
        fractal_visualizer,
        FractalParams
    )
    FRACTAL_VISUALIZER_AVAILABLE = True
    logger.info("Fractal visualizer available")
except ImportError:
    FRACTAL_VISUALIZER_AVAILABLE = False
    logger.warning("Fractal visualizer not available")

class RitualType:
    """Ritual type definitions"""
    SYNCHRONICITY = "synchronicity"
    DECAY_WITH_MEMORY = "decay_with_memory"
    FIELD_EXPANSION = "field_expansion"
    VOID_MEDITATION = "void_meditation"
    GLIFO_ACTIVATION = "glifo_activation"
    COHERENCE_CALIBRATION = "coherence_calibration"
    SYMBOLIC_RESET = "symbolic_reset"
    AURA_MAPPING = "aura_mapping"

class RitualElement:
    """Ritual element definitions"""
    INCENSE = "incense"
    SOUND = "sound"
    BREATH = "breath"
    MOVEMENT = "movement"
    TEXT = "text"
    VISUALIZATION = "visualization"
    OBJECT = "object"
    SILENCE = "silence"
    LIGHT = "light"
    WATER = "water"

class RitualTrigger:
    """Ritual trigger types"""
    MANUAL = "manual"
    TIME = "time"
    LOCATION = "location"
    SENSOR = "sensor"
    VOICE = "voice"
    STATE = "state"
    PATTERN = "pattern"

class Ritual:
    """
    Ritual definition
    
    Represents a structured sequence of symbolic actions with defined
    triggers, elements, and expected outcomes.
    """
    
    def __init__(self, name: str, ritual_type: str = RitualType.SYNCHRONICITY):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = ""
        self.ritual_type = ritual_type
        self.elements = []
        self.triggers = []
        self.intensity = 5  # 1-10 scale
        self.duration = 10  # minutes
        self.location = None
        self.time_window = None
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.last_performed = None
        self.performance_count = 0
        self.glifo_id = None
        self.aura_state = {}
        self.notes = ""
    
    def add_element(self, element_type: str, description: str, 
                   duration: Optional[int] = None, intensity: Optional[int] = None) -> Dict[str, Any]:
        """Add an element to the ritual"""
        element = {
            "id": str(uuid.uuid4()),
            "type": element_type,
            "description": description,
            "duration": duration,  # in seconds, or None for indefinite
            "intensity": intensity or self.intensity,
            "order": len(self.elements) + 1
        }
        
        self.elements.append(element)
        self.updated_at = datetime.now().isoformat()
        
        return element
    
    def add_trigger(self, trigger_type: str, condition: str, 
                   parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add a trigger to the ritual"""
        trigger = {
            "id": str(uuid.uuid4()),
            "type": trigger_type,
            "condition": condition,
            "parameters": parameters or {},
            "active": True
        }
        
        self.triggers.append(trigger)
        self.updated_at = datetime.now().isoformat()
        
        return trigger
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ritual to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "ritual_type": self.ritual_type,
            "elements": self.elements,
            "triggers": self.triggers,
            "intensity": self.intensity,
            "duration": self.duration,
            "location": self.location,
            "time_window": self.time_window,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_performed": self.last_performed,
            "performance_count": self.performance_count,
            "glifo_id": self.glifo_id,
            "aura_state": self.aura_state,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Ritual':
        """Create ritual from dictionary"""
        ritual = cls(name=data.get("name", "Unnamed Ritual"))
        
        # Copy all fields
        ritual.id = data.get("id", ritual.id)
        ritual.description = data.get("description", "")
        ritual.ritual_type = data.get("ritual_type", RitualType.SYNCHRONICITY)
        ritual.elements = data.get("elements", [])
        ritual.triggers = data.get("triggers", [])
        ritual.intensity = data.get("intensity", 5)
        ritual.duration = data.get("duration", 10)
        ritual.location = data.get("location")
        ritual.time_window = data.get("time_window")
        ritual.created_at = data.get("created_at", ritual.created_at)
        ritual.updated_at = data.get("updated_at", ritual.updated_at)
        ritual.last_performed = data.get("last_performed")
        ritual.performance_count = data.get("performance_count", 0)
        ritual.glifo_id = data.get("glifo_id")
        ritual.aura_state = data.get("aura_state", {})
        ritual.notes = data.get("notes", "")
        
        return ritual

class RitualLog:
    """
    Ritual performance log
    
    Records details of a ritual performance, including timing,
    observations, and measured effects.
    """
    
    def __init__(self, ritual_id: str, ritual_name: str):
        self.id = str(uuid.uuid4())
        self.ritual_id = ritual_id
        self.ritual_name = ritual_name
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.duration = None
        self.location = None
        self.elements_performed = []
        self.observations = []
        self.measured_effects = {}
        self.glifo_id = None
        self.aura_state_before = {}
        self.aura_state_after = {}
        self.cognitive_resonance = None
        self.notes = ""
        self.status = "in_progress"  # in_progress, completed, interrupted
    
    def add_observation(self, timestamp: Optional[datetime] = None, 
                      text: str = "", data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add an observation during the ritual"""
        observation = {
            "id": str(uuid.uuid4()),
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "text": text,
            "data": data or {}
        }
        
        self.observations.append(observation)
        return observation
    
    def complete(self, notes: str = "", measured_effects: Optional[Dict[str, Any]] = None,
               aura_state: Optional[Dict[str, Any]] = None, cognitive_resonance: Optional[float] = None):
        """Mark the ritual as complete and record final measurements"""
        self.end_time = datetime.now().isoformat()
        self.duration = (datetime.fromisoformat(self.end_time) - 
                        datetime.fromisoformat(self.start_time)).total_seconds() / 60  # minutes
        self.notes = notes
        self.measured_effects = measured_effects or {}
        self.aura_state_after = aura_state or {}
        self.cognitive_resonance = cognitive_resonance
        self.status = "completed"
        
        # Generate a Glifo ID based on the ritual performance
        self.glifo_id = self._generate_glifo_id()
    
    def _generate_glifo_id(self) -> str:
        """Generate a unique Glifo ID for this ritual performance"""
        # Create a deterministic but unique ID based on ritual parameters
        base = f"{self.ritual_id}_{self.start_time}"
        if self.cognitive_resonance is not None:
            base += f"_{self.cognitive_resonance:.4f}"
        
        # Use observations if available
        if self.observations:
            sample_obs = "_".join(obs.get("text", "")[:10] for obs in self.observations[:2])
            base += f"_{sample_obs}"
        
        # Create a SHA hash of the base string
        import hashlib
        glifo_hash = hashlib.sha256(base.encode()).hexdigest()[:12]
        
        # Format as a GID (Glifo ID)
        return f"GID-{glifo_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log to dictionary"""
        return {
            "id": self.id,
            "ritual_id": self.ritual_id,
            "ritual_name": self.ritual_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "location": self.location,
            "elements_performed": self.elements_performed,
            "observations": self.observations,
            "measured_effects": self.measured_effects,
            "glifo_id": self.glifo_id,
            "aura_state_before": self.aura_state_before,
            "aura_state_after": self.aura_state_after,
            "cognitive_resonance": self.cognitive_resonance,
            "notes": self.notes,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RitualLog':
        """Create log from dictionary"""
        log = cls(
            ritual_id=data.get("ritual_id", "unknown"),
            ritual_name=data.get("ritual_name", "Unknown Ritual")
        )
        
        # Copy all fields
        log.id = data.get("id", log.id)
        log.start_time = data.get("start_time", log.start_time)
        log.end_time = data.get("end_time")
        log.duration = data.get("duration")
        log.location = data.get("location")
        log.elements_performed = data.get("elements_performed", [])
        log.observations = data.get("observations", [])
        log.measured_effects = data.get("measured_effects", {})
        log.glifo_id = data.get("glifo_id")
        log.aura_state_before = data.get("aura_state_before", {})
        log.aura_state_after = data.get("aura_state_after", {})
        log.cognitive_resonance = data.get("cognitive_resonance")
        log.notes = data.get("notes", "")
        log.status = data.get("status", "completed")
        
        return log

class RitualEngine:
    """
    WiltonOS Ritual Engine
    
    Core system for defining, performing, and tracking symbolic rituals
    with full integration into the WiltonOS ecosystem.
    """
    
    def __init__(self):
        """Initialize the ritual engine"""
        # Storage for rituals and logs
        self._rituals = {}
        self._logs = []
        
        # Active ritual performance
        self.active_ritual = None
        self.active_log = None
        
        # Try to load existing data
        self._load_data()
        
        logger.info("Ritual Engine initialized")
    
    def _load_data(self):
        """Load rituals and logs from storage"""
        # Create storage directories if they don't exist
        os.makedirs("state/rituals", exist_ok=True)
        os.makedirs("state/logs", exist_ok=True)
        
        # Load rituals
        if os.path.exists("state/rituals.json"):
            try:
                with open("state/rituals.json", "r") as f:
                    rituals_data = json.load(f)
                
                for ritual_data in rituals_data:
                    ritual = Ritual.from_dict(ritual_data)
                    self._rituals[ritual.id] = ritual
                
                logger.info(f"Loaded {len(self._rituals)} rituals")
            except Exception as e:
                logger.error(f"Error loading rituals: {str(e)}")
        
        # Load logs
        if os.path.exists("state/ritual_logs.json"):
            try:
                with open("state/ritual_logs.json", "r") as f:
                    logs_data = json.load(f)
                
                for log_data in logs_data:
                    log = RitualLog.from_dict(log_data)
                    self._logs.append(log)
                
                logger.info(f"Loaded {len(self._logs)} ritual logs")
            except Exception as e:
                logger.error(f"Error loading ritual logs: {str(e)}")
    
    def _save_data(self):
        """Save rituals and logs to storage"""
        # Save rituals
        try:
            rituals_data = [ritual.to_dict() for ritual in self._rituals.values()]
            with open("state/rituals.json", "w") as f:
                json.dump(rituals_data, f, indent=2)
            
            logger.info(f"Saved {len(rituals_data)} rituals")
        except Exception as e:
            logger.error(f"Error saving rituals: {str(e)}")
        
        # Save logs
        try:
            logs_data = [log.to_dict() for log in self._logs]
            with open("state/ritual_logs.json", "w") as f:
                json.dump(logs_data, f, indent=2)
            
            logger.info(f"Saved {len(logs_data)} ritual logs")
        except Exception as e:
            logger.error(f"Error saving ritual logs: {str(e)}")
    
    def create_ritual(self, name: str, ritual_type: str = RitualType.SYNCHRONICITY) -> Ritual:
        """Create a new ritual"""
        ritual = Ritual(name, ritual_type)
        self._rituals[ritual.id] = ritual
        self._save_data()
        
        logger.info(f"Created ritual: {ritual.name} ({ritual.id})")
        return ritual
    
    def update_ritual(self, ritual_id: str, **kwargs) -> Optional[Ritual]:
        """Update an existing ritual"""
        if ritual_id not in self._rituals:
            logger.warning(f"Ritual {ritual_id} not found")
            return None
        
        ritual = self._rituals[ritual_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(ritual, key):
                setattr(ritual, key, value)
        
        ritual.updated_at = datetime.now().isoformat()
        self._save_data()
        
        logger.info(f"Updated ritual: {ritual.name} ({ritual.id})")
        return ritual
    
    def delete_ritual(self, ritual_id: str) -> bool:
        """Delete a ritual"""
        if ritual_id not in self._rituals:
            logger.warning(f"Ritual {ritual_id} not found")
            return False
        
        ritual_name = self._rituals[ritual_id].name
        del self._rituals[ritual_id]
        self._save_data()
        
        logger.info(f"Deleted ritual: {ritual_name} ({ritual_id})")
        return True
    
    def get_ritual(self, ritual_id: str) -> Optional[Ritual]:
        """Get a ritual by ID"""
        return self._rituals.get(ritual_id)
    
    def get_rituals(self, ritual_type: Optional[str] = None) -> List[Ritual]:
        """Get all rituals, optionally filtered by type"""
        if ritual_type:
            return [r for r in self._rituals.values() if r.ritual_type == ritual_type]
        return list(self._rituals.values())
    
    def start_ritual(self, ritual_id: str) -> Optional[RitualLog]:
        """Start performing a ritual"""
        if self.active_ritual:
            logger.warning("Another ritual is already in progress")
            return None
        
        ritual = self.get_ritual(ritual_id)
        if not ritual:
            logger.warning(f"Ritual {ritual_id} not found")
            return None
        
        # Create a new log
        log = RitualLog(ritual_id, ritual.name)
        
        # Record initial state
        log.aura_state_before = self._get_current_aura_state()
        
        # Try to get location
        try:
            # This is a placeholder - in a real implementation, 
            # this would use actual location services
            geolocator = Nominatim(user_agent="wiltonos_ritual_engine")
            location = geolocator.geocode("São Paulo, Brazil")
            if location:
                log.location = {
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                    "address": location.address
                }
        except:
            # Geolocation is optional, continue if it fails
            pass
        
        # Set as active
        self.active_ritual = ritual
        self.active_log = log
        
        # Update ritual record
        ritual.last_performed = log.start_time
        ritual.performance_count += 1
        
        logger.info(f"Started ritual: {ritual.name} ({ritual.id})")
        return log
    
    def add_ritual_observation(self, text: str, data: Optional[Dict[str, Any]] = None) -> bool:
        """Add an observation to the active ritual log"""
        if not self.active_log:
            logger.warning("No active ritual to add observation to")
            return False
        
        self.active_log.add_observation(text=text, data=data)
        logger.info(f"Added observation to ritual {self.active_ritual.name}: {text[:30]}...")
        return True
    
    def record_element_performed(self, element_id: str, notes: str = "") -> bool:
        """Record that a ritual element was performed"""
        if not self.active_ritual or not self.active_log:
            logger.warning("No active ritual")
            return False
        
        # Find the element
        element = next((e for e in self.active_ritual.elements if e["id"] == element_id), None)
        if not element:
            logger.warning(f"Element {element_id} not found in ritual {self.active_ritual.id}")
            return False
        
        # Record the performance
        element_record = {
            "id": element_id,
            "type": element["type"],
            "description": element["description"],
            "timestamp": datetime.now().isoformat(),
            "notes": notes
        }
        
        self.active_log.elements_performed.append(element_record)
        logger.info(f"Recorded element performance: {element['type']}")
        return True
    
    def complete_ritual(self, notes: str = "", cognitive_resonance: Optional[float] = None) -> RitualLog:
        """Complete the active ritual and record final measurements"""
        if not self.active_ritual or not self.active_log:
            logger.warning("No active ritual to complete")
            return None
        
        # Measure final state
        aura_state = self._get_current_aura_state()
        
        # Calculate effects if not provided
        measured_effects = self._calculate_ritual_effects(
            self.active_ritual, 
            self.active_log.aura_state_before,
            aura_state
        )
        
        # Complete the log
        self.active_log.complete(
            notes=notes,
            measured_effects=measured_effects,
            aura_state=aura_state,
            cognitive_resonance=cognitive_resonance
        )
        
        # Update ritual with Glifo ID if generated
        if self.active_log.glifo_id and not self.active_ritual.glifo_id:
            self.active_ritual.glifo_id = self.active_log.glifo_id
        
        # Add to logs and save
        completed_log = self.active_log
        self._logs.append(completed_log)
        self._save_data()
        
        # Clear active ritual
        self.active_ritual = None
        self.active_log = None
        
        logger.info(f"Completed ritual: {completed_log.ritual_name}")
        return completed_log
    
    def cancel_ritual(self, reason: str = "") -> bool:
        """Cancel the active ritual"""
        if not self.active_ritual or not self.active_log:
            logger.warning("No active ritual to cancel")
            return False
        
        # Mark log as interrupted
        self.active_log.status = "interrupted"
        self.active_log.end_time = datetime.now().isoformat()
        self.active_log.notes = f"Canceled: {reason}"
        
        # Add to logs and save
        self._logs.append(self.active_log)
        self._save_data()
        
        # Clear active ritual
        ritual_name = self.active_ritual.name
        self.active_ritual = None
        self.active_log = None
        
        logger.info(f"Canceled ritual: {ritual_name}")
        return True
    
    def get_logs(self, ritual_id: Optional[str] = None, 
               limit: int = 100, status: Optional[str] = None) -> List[RitualLog]:
        """Get ritual logs, optionally filtered"""
        filtered_logs = self._logs
        
        if ritual_id:
            filtered_logs = [log for log in filtered_logs if log.ritual_id == ritual_id]
        
        if status:
            filtered_logs = [log for log in filtered_logs if log.status == status]
        
        # Sort by start time (newest first)
        filtered_logs.sort(key=lambda log: log.start_time, reverse=True)
        
        return filtered_logs[:limit]
    
    def get_log(self, log_id: str) -> Optional[RitualLog]:
        """Get a specific log by ID"""
        return next((log for log in self._logs if log.id == log_id), None)
    
    def _get_current_aura_state(self) -> Dict[str, Any]:
        """
        Get the current aura state of the system
        
        This is a placeholder implementation. In a real system, this would
        integrate with various sensors, APIs, or other data sources to
        capture the current state of the user and environment.
        """
        # Placeholder implementation with some randomness
        # In a real implementation, this would use actual measurements
        return {
            "timestamp": datetime.now().isoformat(),
            "cognitive_resonance": round(np.random.uniform(0.5, 1.0), 2),
            "emotional_state": {
                "valence": round(np.random.uniform(-1.0, 1.0), 2),
                "arousal": round(np.random.uniform(0.0, 1.0), 2),
                "dominance": round(np.random.uniform(0.0, 1.0), 2)
            },
            "physical_state": {
                "energy": round(np.random.uniform(0.3, 1.0), 2),
                "relaxation": round(np.random.uniform(0.3, 1.0), 2)
            },
            "environmental_factors": {
                "noise_level": round(np.random.uniform(0.0, 0.5), 2),
                "light_level": round(np.random.uniform(0.2, 0.8), 2),
                "air_quality": round(np.random.uniform(0.5, 1.0), 2)
            }
        }
    
    def _calculate_ritual_effects(self, ritual: Ritual, 
                              before_state: Dict[str, Any],
                              after_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the effects of a ritual by comparing before and after states
        
        This is a placeholder implementation. In a real system, this would
        use more sophisticated analysis methods.
        """
        effects = {
            "cognitive_resonance_delta": 0,
            "emotional_state_deltas": {},
            "physical_state_deltas": {},
            "environmental_deltas": {}
        }
        
        # Calculate cognitive resonance delta
        if "cognitive_resonance" in before_state and "cognitive_resonance" in after_state:
            effects["cognitive_resonance_delta"] = round(
                after_state["cognitive_resonance"] - before_state["cognitive_resonance"], 
                2
            )
        
        # Calculate emotional state deltas
        if "emotional_state" in before_state and "emotional_state" in after_state:
            for key in before_state["emotional_state"]:
                if key in after_state["emotional_state"]:
                    effects["emotional_state_deltas"][key] = round(
                        after_state["emotional_state"][key] - before_state["emotional_state"][key],
                        2
                    )
        
        # Calculate physical state deltas
        if "physical_state" in before_state and "physical_state" in after_state:
            for key in before_state["physical_state"]:
                if key in after_state["physical_state"]:
                    effects["physical_state_deltas"][key] = round(
                        after_state["physical_state"][key] - before_state["physical_state"][key],
                        2
                    )
        
        # Calculate environmental deltas
        if "environmental_factors" in before_state and "environmental_factors" in after_state:
            for key in before_state["environmental_factors"]:
                if key in after_state["environmental_factors"]:
                    effects["environmental_deltas"][key] = round(
                        after_state["environmental_factors"][key] - before_state["environmental_factors"][key],
                        2
                    )
        
        # Calculate a ritual effectiveness score
        # This is a simplified placeholder - would be more sophisticated in practice
        cr_weight = 0.4
        emotional_weight = 0.3
        physical_weight = 0.2
        environmental_weight = 0.1
        
        cr_score = abs(effects["cognitive_resonance_delta"]) * 10  # Scale to 0-10
        
        emotional_scores = [abs(v) * 10 for v in effects["emotional_state_deltas"].values()]
        emotional_score = sum(emotional_scores) / len(emotional_scores) if emotional_scores else 0
        
        physical_scores = [abs(v) * 10 for v in effects["physical_state_deltas"].values()]
        physical_score = sum(physical_scores) / len(physical_scores) if physical_scores else 0
        
        environmental_scores = [abs(v) * 10 for v in effects["environmental_deltas"].values()]
        environmental_score = sum(environmental_scores) / len(environmental_scores) if environmental_scores else 0
        
        effectiveness = (
            cr_score * cr_weight +
            emotional_score * emotional_weight +
            physical_score * physical_weight +
            environmental_score * environmental_weight
        )
        
        effects["effectiveness"] = round(min(effectiveness, 10), 2)  # Scale to 0-10
        
        return effects
    
    def check_triggers(self) -> List[str]:
        """
        Check all ritual triggers to see if any rituals should be activated
        
        Returns a list of ritual IDs that have triggered.
        """
        triggered = []
        
        # Skip if there's already an active ritual
        if self.active_ritual:
            return triggered
        
        current_time = datetime.now()
        
        for ritual_id, ritual in self._rituals.items():
            for trigger in ritual.triggers:
                if not trigger.get("active", True):
                    continue
                
                trigger_type = trigger.get("type")
                condition = trigger.get("condition", "")
                parameters = trigger.get("parameters", {})
                
                # Check time-based triggers
                if trigger_type == RitualTrigger.TIME:
                    if self._check_time_trigger(condition, parameters, current_time):
                        triggered.append(ritual_id)
                
                # Check location-based triggers (placeholder)
                elif trigger_type == RitualTrigger.LOCATION:
                    # This would use actual location services in a real implementation
                    pass
                
                # Check voice triggers
                elif trigger_type == RitualTrigger.VOICE and WHISPER_AVAILABLE:
                    # This would integrate with Whisper in a real implementation
                    pass
                
                # Check state triggers
                elif trigger_type == RitualTrigger.STATE:
                    if self._check_state_trigger(condition, parameters):
                        triggered.append(ritual_id)
        
        return triggered
    
    def _check_time_trigger(self, condition: str, parameters: Dict[str, Any],
                          current_time: datetime) -> bool:
        """Check if a time-based trigger condition is met"""
        if condition == "daily":
            # Check for daily time trigger
            target_time = parameters.get("time")
            if target_time:
                # Parse HH:MM format
                try:
                    hours, minutes = map(int, target_time.split(":"))
                    if current_time.hour == hours and current_time.minute == minutes:
                        return True
                except:
                    pass
        
        elif condition == "window":
            # Check for time window trigger
            start_time = parameters.get("start_time")
            end_time = parameters.get("end_time")
            if start_time and end_time:
                try:
                    start_hours, start_minutes = map(int, start_time.split(":"))
                    end_hours, end_minutes = map(int, end_time.split(":"))
                    
                    start = current_time.replace(hour=start_hours, minute=start_minutes)
                    end = current_time.replace(hour=end_hours, minute=end_minutes)
                    
                    if start <= current_time <= end:
                        return True
                except:
                    pass
        
        return False
    
    def _check_state_trigger(self, condition: str, parameters: Dict[str, Any]) -> bool:
        """Check if a state-based trigger condition is met"""
        if condition == "cognitive_resonance":
            # Check cognitive resonance threshold
            threshold = parameters.get("threshold", 0.8)
            current_state = self._get_current_aura_state()
            current_cr = current_state.get("cognitive_resonance", 0)
            
            if current_cr >= threshold:
                return True
        
        return False
    
    def get_ritual_analytics(self, ritual_id: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics for rituals"""
        # Get relevant logs
        logs = self.get_logs(ritual_id=ritual_id)
        
        if not logs:
            return {
                "total_rituals": 0,
                "total_performances": 0,
                "completion_rate": 0,
                "average_effectiveness": 0,
                "average_cognitive_resonance": 0,
                "ritual_counts": {},
                "time_distribution": {}
            }
        
        # Calculate analytics
        total_performances = len(logs)
        completed_performances = sum(1 for log in logs if log.status == "completed")
        completion_rate = round(completed_performances / total_performances * 100, 1) if total_performances > 0 else 0
        
        # Effectiveness and cognitive resonance
        effectiveness_values = []
        cr_values = []
        
        for log in logs:
            if log.status == "completed":
                if log.measured_effects and "effectiveness" in log.measured_effects:
                    effectiveness_values.append(log.measured_effects["effectiveness"])
                
                if log.cognitive_resonance is not None:
                    cr_values.append(log.cognitive_resonance)
        
        avg_effectiveness = round(sum(effectiveness_values) / len(effectiveness_values), 2) if effectiveness_values else 0
        avg_cr = round(sum(cr_values) / len(cr_values), 2) if cr_values else 0
        
        # Ritual type counts
        ritual_counts = {}
        for log in logs:
            ritual = self.get_ritual(log.ritual_id)
            if ritual:
                ritual_type = ritual.ritual_type
                ritual_counts[ritual_type] = ritual_counts.get(ritual_type, 0) + 1
        
        # Time distribution
        time_distribution = {
            "morning": 0,
            "afternoon": 0,
            "evening": 0,
            "night": 0
        }
        
        for log in logs:
            try:
                start_time = datetime.fromisoformat(log.start_time)
                hour = start_time.hour
                
                if 5 <= hour < 12:
                    time_distribution["morning"] += 1
                elif 12 <= hour < 17:
                    time_distribution["afternoon"] += 1
                elif 17 <= hour < 21:
                    time_distribution["evening"] += 1
                else:
                    time_distribution["night"] += 1
            except:
                pass
        
        return {
            "total_rituals": len(set(log.ritual_id for log in logs)),
            "total_performances": total_performances,
            "completion_rate": completion_rate,
            "average_effectiveness": avg_effectiveness,
            "average_cognitive_resonance": avg_cr,
            "ritual_counts": ritual_counts,
            "time_distribution": time_distribution
        }

class RitualInterface:
    """
    Ritual Engine Interface
    
    Streamlit interface for creating, performing, and analyzing rituals
    """
    
    def __init__(self):
        """Initialize the interface"""
        # Initialize the ritual engine
        self.engine = RitualEngine()
        
        # Initialize session state variables
        if 'current_ritual_id' not in st.session_state:
            st.session_state.current_ritual_id = None
        
        if 'current_log_id' not in st.session_state:
            st.session_state.current_log_id = None
        
        if 'ritual_timer_active' not in st.session_state:
            st.session_state.ritual_timer_active = False
        
        if 'ritual_timer_start' not in st.session_state:
            st.session_state.ritual_timer_start = None
        
        if 'ritual_step_index' not in st.session_state:
            st.session_state.ritual_step_index = 0
        
        if 'show_ritual_results' not in st.session_state:
            st.session_state.show_ritual_results = False
        
        if 'last_completed_log' not in st.session_state:
            st.session_state.last_completed_log = None
        
        logger.info("Ritual Interface initialized")
    
    def _format_duration(self, seconds: float) -> str:
        """Format seconds as MM:SS"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def render_ritual_creator(self):
        """Render the ritual creation interface"""
        st.markdown("### 🪄 Create New Ritual")
        
        with st.form("ritual_creator_form"):
            # Basic ritual information
            name = st.text_input("Ritual Name", value="New Ritual")
            
            col1, col2 = st.columns(2)
            
            with col1:
                ritual_type = st.selectbox(
                    "Ritual Type",
                    options=[
                        RitualType.SYNCHRONICITY,
                        RitualType.DECAY_WITH_MEMORY,
                        RitualType.FIELD_EXPANSION,
                        RitualType.VOID_MEDITATION,
                        RitualType.GLIFO_ACTIVATION,
                        RitualType.COHERENCE_CALIBRATION,
                        RitualType.SYMBOLIC_RESET,
                        RitualType.AURA_MAPPING
                    ],
                    format_func=lambda x: x.replace("_", " ").title()
                )
            
            with col2:
                intensity = st.slider("Intensity", 1, 10, 5)
                duration = st.slider("Duration (minutes)", 1, 60, 10)
            
            description = st.text_area("Description", value="")
            
            submitted = st.form_submit_button("Create Ritual")
            
            if submitted:
                if name:
                    ritual = self.engine.create_ritual(name, ritual_type)
                    ritual.description = description
                    ritual.intensity = intensity
                    ritual.duration = duration
                    
                    st.success(f"Ritual '{name}' created")
                    
                    # Set as current ritual
                    st.session_state.current_ritual_id = ritual.id
                    
                    # Force refresh
                    st.experimental_rerun()
                else:
                    st.error("Ritual name cannot be empty")
    
    def render_ritual_editor(self):
        """Render the ritual editing interface"""
        if not st.session_state.current_ritual_id:
            st.info("Select a ritual to edit")
            
            # Show ritual selector
            rituals = self.engine.get_rituals()
            if not rituals:
                st.warning("No rituals found. Create one first.")
                return
            
            ritual_options = {r.name: r.id for r in rituals}
            selected_name = st.selectbox(
                "Select Ritual",
                options=list(ritual_options.keys())
            )
            
            if st.button("Edit Selected Ritual"):
                st.session_state.current_ritual_id = ritual_options[selected_name]
                st.experimental_rerun()
            
            return
        
        # Get the current ritual
        ritual = self.engine.get_ritual(st.session_state.current_ritual_id)
        if not ritual:
            st.error("Ritual not found")
            st.session_state.current_ritual_id = None
            return
        
        st.markdown(f"### ✏️ Editing: {ritual.name}")
        
        # Main settings
        with st.expander("Basic Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                new_name = st.text_input("Ritual Name", value=ritual.name)
                
                new_type = st.selectbox(
                    "Ritual Type",
                    options=[
                        RitualType.SYNCHRONICITY,
                        RitualType.DECAY_WITH_MEMORY,
                        RitualType.FIELD_EXPANSION,
                        RitualType.VOID_MEDITATION,
                        RitualType.GLIFO_ACTIVATION,
                        RitualType.COHERENCE_CALIBRATION,
                        RitualType.SYMBOLIC_RESET,
                        RitualType.AURA_MAPPING
                    ],
                    index=[
                        RitualType.SYNCHRONICITY,
                        RitualType.DECAY_WITH_MEMORY,
                        RitualType.FIELD_EXPANSION,
                        RitualType.VOID_MEDITATION,
                        RitualType.GLIFO_ACTIVATION,
                        RitualType.COHERENCE_CALIBRATION,
                        RitualType.SYMBOLIC_RESET,
                        RitualType.AURA_MAPPING
                    ].index(ritual.ritual_type),
                    format_func=lambda x: x.replace("_", " ").title()
                )
            
            with col2:
                new_intensity = st.slider("Intensity", 1, 10, ritual.intensity)
                new_duration = st.slider("Duration (minutes)", 1, 60, ritual.duration)
            
            new_description = st.text_area("Description", value=ritual.description)
            
            if st.button("Update Basic Settings"):
                # Update ritual
                self.engine.update_ritual(
                    ritual.id,
                    name=new_name,
                    ritual_type=new_type,
                    intensity=new_intensity,
                    duration=new_duration,
                    description=new_description
                )
                
                st.success("Ritual updated")
                st.experimental_rerun()
        
        # Elements
        with st.expander("Ritual Elements", expanded=True):
            st.markdown("#### Current Elements")
            
            if not ritual.elements:
                st.info("No elements added yet")
            else:
                for i, element in enumerate(ritual.elements):
                    with st.container():
                        col1, col2, col3 = st.columns([1, 3, 1])
                        
                        with col1:
                            st.markdown(f"**{i+1}. {element['type'].title()}**")
                        
                        with col2:
                            st.markdown(element['description'])
                        
                        with col3:
                            if st.button("Remove", key=f"remove_element_{i}"):
                                ritual.elements.pop(i)
                                self.engine._save_data()
                                st.experimental_rerun()
                    
                    if i < len(ritual.elements) - 1:
                        st.markdown("---")
            
            st.markdown("#### Add Element")
            
            with st.form("add_element_form"):
                element_type = st.selectbox(
                    "Element Type",
                    options=[
                        RitualElement.INCENSE,
                        RitualElement.SOUND,
                        RitualElement.BREATH,
                        RitualElement.MOVEMENT,
                        RitualElement.TEXT,
                        RitualElement.VISUALIZATION,
                        RitualElement.OBJECT,
                        RitualElement.SILENCE,
                        RitualElement.LIGHT,
                        RitualElement.WATER
                    ],
                    format_func=lambda x: x.title()
                )
                
                element_description = st.text_area("Description")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    element_duration = st.number_input(
                        "Duration (seconds, 0 for indefinite)",
                        min_value=0,
                        value=60
                    )
                
                with col2:
                    element_intensity = st.slider(
                        "Intensity",
                        1, 10, ritual.intensity
                    )
                
                submitted = st.form_submit_button("Add Element")
                
                if submitted:
                    if element_description:
                        # Add element
                        ritual.add_element(
                            element_type=element_type,
                            description=element_description,
                            duration=element_duration if element_duration > 0 else None,
                            intensity=element_intensity
                        )
                        
                        # Save
                        self.engine._save_data()
                        
                        st.success("Element added")
                        st.experimental_rerun()
                    else:
                        st.error("Element description cannot be empty")
        
        # Triggers
        with st.expander("Ritual Triggers"):
            st.markdown("#### Current Triggers")
            
            if not ritual.triggers:
                st.info("No triggers added yet")
            else:
                for i, trigger in enumerate(ritual.triggers):
                    with st.container():
                        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{trigger['type'].title()}**")
                        
                        with col2:
                            st.markdown(trigger['condition'])
                        
                        with col3:
                            active = trigger.get('active', True)
                            new_active = st.checkbox("Active", value=active, key=f"trigger_active_{i}")
                            if new_active != active:
                                trigger['active'] = new_active
                                self.engine._save_data()
                        
                        with col4:
                            if st.button("Remove", key=f"remove_trigger_{i}"):
                                ritual.triggers.pop(i)
                                self.engine._save_data()
                                st.experimental_rerun()
                    
                    if i < len(ritual.triggers) - 1:
                        st.markdown("---")
            
            st.markdown("#### Add Trigger")
            
            with st.form("add_trigger_form"):
                trigger_type = st.selectbox(
                    "Trigger Type",
                    options=[
                        RitualTrigger.MANUAL,
                        RitualTrigger.TIME,
                        RitualTrigger.LOCATION,
                        RitualTrigger.VOICE,
                        RitualTrigger.STATE,
                        RitualTrigger.PATTERN
                    ],
                    format_func=lambda x: x.title()
                )
                
                # Dynamic parameters based on trigger type
                condition = ""
                parameters = {}
                
                if trigger_type == RitualTrigger.TIME:
                    condition = st.selectbox(
                        "Time Condition",
                        options=["daily", "window"]
                    )
                    
                    if condition == "daily":
                        parameters["time"] = st.text_input("Time (HH:MM)", value="08:00")
                    elif condition == "window":
                        col1, col2 = st.columns(2)
                        with col1:
                            parameters["start_time"] = st.text_input("Start Time (HH:MM)", value="08:00")
                        with col2:
                            parameters["end_time"] = st.text_input("End Time (HH:MM)", value="10:00")
                
                elif trigger_type == RitualTrigger.LOCATION:
                    condition = st.text_input("Location Name")
                    parameters["radius"] = st.number_input("Radius (meters)", min_value=10, value=100)
                
                elif trigger_type == RitualTrigger.VOICE:
                    condition = st.text_input("Trigger Phrase")
                
                elif trigger_type == RitualTrigger.STATE:
                    condition = "cognitive_resonance"
                    parameters["threshold"] = st.slider("Threshold", 0.0, 1.0, 0.8, step=0.05)
                
                else:  # MANUAL or PATTERN
                    condition = st.text_input("Condition Description")
                
                submitted = st.form_submit_button("Add Trigger")
                
                if submitted:
                    if condition:
                        # Add trigger
                        ritual.add_trigger(
                            trigger_type=trigger_type,
                            condition=condition,
                            parameters=parameters
                        )
                        
                        # Save
                        self.engine._save_data()
                        
                        st.success("Trigger added")
                        st.experimental_rerun()
                    else:
                        st.error("Condition cannot be empty")
        
        # Delete ritual button
        st.divider()
        
        if st.button("⚠️ Delete Ritual", key="delete_ritual"):
            if st.session_state.current_ritual_id:
                self.engine.delete_ritual(st.session_state.current_ritual_id)
                st.session_state.current_ritual_id = None
                st.success("Ritual deleted")
                st.experimental_rerun()
    
    def render_ritual_performer(self):
        """Render the ritual performance interface"""
        # Check if a ritual performance is active
        if self.engine.active_ritual and self.engine.active_log:
            self._render_active_ritual()
            return
        
        # Check if we're showing results
        if st.session_state.show_ritual_results and st.session_state.last_completed_log:
            self._render_ritual_results()
            return
        
        # Default view - ritual selector
        st.markdown("### 🌟 Perform Ritual")
        
        # Get available rituals
        rituals = self.engine.get_rituals()
        if not rituals:
            st.warning("No rituals found. Create one first.")
            return
        
        # Display rituals as cards
        st.markdown("#### Select a Ritual to Perform")
        
        # Create columns for ritual cards
        cols = st.columns(2)
        
        for i, ritual in enumerate(rituals):
            with cols[i % 2]:
                # Ritual card
                card_html = f"""
                <div style="background-color: rgba(30, 40, 50, 0.8); border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #6E44FF;">
                    <div style="font-weight: 500; font-size: 1.1rem; margin-bottom: 5px;">{ritual.name}</div>
                    <div style="font-size: 0.9rem; color: #ddd; margin-bottom: 10px;">{ritual.description[:100]}...</div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #aaa;">
                        <div>Type: {ritual.ritual_type.replace('_', ' ').title()}</div>
                        <div>Intensity: {ritual.intensity}/10</div>
                        <div>Duration: {ritual.duration} min</div>
                    </div>
                </div>
                """
                
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Start button
                if st.button("Begin Ritual", key=f"start_ritual_{ritual.id}"):
                    # Start the ritual
                    log = self.engine.start_ritual(ritual.id)
                    if log:
                        st.session_state.ritual_timer_active = True
                        st.session_state.ritual_timer_start = datetime.now()
                        st.session_state.ritual_step_index = 0
                        st.success(f"Started ritual: {ritual.name}")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to start ritual")
        
        # Check for triggered rituals
        triggered = self.engine.check_triggers()
        if triggered:
            st.markdown("#### Triggered Rituals")
            
            for ritual_id in triggered:
                ritual = self.engine.get_ritual(ritual_id)
                if ritual:
                    st.markdown(f"""
                    <div style="background-color: rgba(30, 40, 50, 0.8); border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #FFB74D;">
                        <div style="display: flex; align-items: center;">
                            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #FFB74D; margin-right: 8px; animation: pulse 2s infinite;"></div>
                            <div style="font-weight: 500; font-size: 1.1rem;">Ritual Triggered: {ritual.name}</div>
                        </div>
                        <div style="margin-top: 10px; font-size: 0.9rem;">
                            {ritual.description[:100]}...
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Accept", key=f"accept_trigger_{ritual_id}"):
                            log = self.engine.start_ritual(ritual_id)
                            if log:
                                st.session_state.ritual_timer_active = True
                                st.session_state.ritual_timer_start = datetime.now()
                                st.session_state.ritual_step_index = 0
                                st.success(f"Started ritual: {ritual.name}")
                                st.experimental_rerun()
                            else:
                                st.error("Failed to start ritual")
                    
                    with col2:
                        if st.button("Ignore", key=f"ignore_trigger_{ritual_id}"):
                            st.info(f"Ignored ritual trigger: {ritual.name}")
    
    def _render_active_ritual(self):
        """Render the active ritual interface"""
        ritual = self.engine.active_ritual
        log = self.engine.active_log
        
        if not ritual or not log:
            return
        
        # Calculate timer
        start_time = datetime.fromisoformat(log.start_time)
        elapsed = (datetime.now() - start_time).total_seconds()
        remaining = max(0, ritual.duration * 60 - elapsed)
        
        # Header
        st.markdown(f"### 🔮 Performing: {ritual.name}")
        
        # Timer display
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown("#### Elapsed")
            st.markdown(f"<div style='font-size: 2rem; text-align: center;'>{self._format_duration(elapsed)}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Progress")
            progress = min(1.0, elapsed / (ritual.duration * 60))
            
            # Progress ring if available, otherwise use progress bar
            if 'progress_ring' in globals():
                progress_ring(progress * 100, 100, size=150)
            else:
                st.progress(progress)
        
        with col3:
            st.markdown("#### Remaining")
            st.markdown(f"<div style='font-size: 2rem; text-align: center;'>{self._format_duration(remaining)}</div>", unsafe_allow_html=True)
        
        # Element guidance
        st.markdown("### Current Element")
        
        if ritual.elements:
            current_index = min(st.session_state.ritual_step_index, len(ritual.elements) - 1)
            current_element = ritual.elements[current_index]
            
            # Display current element
            element_html = f"""
            <div style="background-color: rgba(30, 40, 50, 0.8); border-radius: 10px; padding: 20px; margin-bottom: 20px; border-left: 4px solid #66BB6A;">
                <div style="font-weight: 500; font-size: 1.2rem; margin-bottom: 10px;">
                    {current_element['type'].title()}
                </div>
                <div style="font-size: 1.1rem; margin-bottom: 15px;">
                    {current_element['description']}
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #aaa;">
                    <div>Step {current_index + 1} of {len(ritual.elements)}</div>
                    {f'<div>Duration: {current_element["duration"]} seconds</div>' if current_element.get("duration") else ''}
                    <div>Intensity: {current_element.get("intensity", ritual.intensity)}/10</div>
                </div>
            </div>
            """
            
            st.markdown(element_html, unsafe_allow_html=True)
            
            # Element navigation
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if current_index > 0:
                    if st.button("⬅️ Previous", key="prev_element"):
                        st.session_state.ritual_step_index -= 1
                        st.experimental_rerun()
            
            with col2:
                if st.button("✓ Complete Element", key="complete_element"):
                    # Record element performance
                    self.engine.record_element_performed(
                        current_element['id'],
                        notes=f"Completed at {datetime.now().strftime('%H:%M:%S')}"
                    )
                    
                    # Move to next element if available
                    if current_index < len(ritual.elements) - 1:
                        st.session_state.ritual_step_index += 1
                        st.success(f"Element completed, moving to next")
                        st.experimental_rerun()
                    else:
                        st.success("All elements completed")
            
            with col3:
                if current_index < len(ritual.elements) - 1:
                    if st.button("➡️ Next", key="next_element"):
                        st.session_state.ritual_step_index += 1
                        st.experimental_rerun()
        else:
            st.info("This ritual has no defined elements. Proceed with your practice.")
        
        # Observations
        st.markdown("### Add Observation")
        
        observation_text = st.text_area("Enter your observation or insight")
        
        if st.button("Record Observation", key="add_observation"):
            if observation_text.strip():
                self.engine.add_ritual_observation(observation_text)
                st.success("Observation recorded")
            else:
                st.warning("Observation cannot be empty")
        
        # Ritual control
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🛑 Cancel Ritual", key="cancel_ritual"):
                if st.session_state.current_ritual_id:
                    reason = "User canceled"
                    if self.engine.cancel_ritual(reason):
                        st.session_state.ritual_timer_active = False
                        st.session_state.ritual_timer_start = None
                        st.session_state.ritual_step_index = 0
                        st.warning("Ritual canceled")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to cancel ritual")
        
        with col2:
            if st.button("✅ Complete Ritual", key="complete_ritual"):
                # Request cognitive resonance rating
                cr_rating = st.slider(
                    "Rate your cognitive resonance (0-1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05
                )
                
                notes = st.text_area("Final Notes (optional)")
                
                if st.button("Confirm Completion", key="confirm_completion"):
                    # Complete the ritual
                    completed_log = self.engine.complete_ritual(
                        notes=notes,
                        cognitive_resonance=cr_rating
                    )
                    
                    if completed_log:
                        st.session_state.ritual_timer_active = False
                        st.session_state.ritual_timer_start = None
                        st.session_state.ritual_step_index = 0
                        st.session_state.show_ritual_results = True
                        st.session_state.last_completed_log = completed_log.id
                        st.success("Ritual completed successfully")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to complete ritual")
    
    def _render_ritual_results(self):
        """Render the ritual results interface"""
        log_id = st.session_state.last_completed_log
        if not log_id:
            st.session_state.show_ritual_results = False
            st.experimental_rerun()
            return
        
        log = self.engine.get_log(log_id)
        if not log:
            st.session_state.show_ritual_results = False
            st.experimental_rerun()
            return
        
        ritual = self.engine.get_ritual(log.ritual_id)
        
        # Header
        st.markdown("### ✨ Ritual Results")
        
        # Basic information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Ritual:** {log.ritual_name}")
            st.markdown(f"**Type:** {ritual.ritual_type.replace('_', ' ').title() if ritual else 'Unknown'}")
            start_time = datetime.fromisoformat(log.start_time)
            end_time = datetime.fromisoformat(log.end_time) if log.end_time else datetime.now()
            st.markdown(f"**Time:** {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%H:%M')}")
            st.markdown(f"**Duration:** {log.duration:.1f} minutes")
        
        with col2:
            if log.cognitive_resonance is not None:
                self._render_cognitive_resonance_gauge(log.cognitive_resonance)
            
            # Display effectiveness if available
            if log.measured_effects and "effectiveness" in log.measured_effects:
                effectiveness = log.measured_effects["effectiveness"]
                st.markdown(f"**Effectiveness:** {effectiveness:.1f}/10")
            
            # Display Glifo ID if available
            if log.glifo_id:
                st.markdown(f"**Glifo ID:** `{log.glifo_id}`")
        
        # Show observations
        if log.observations:
            st.markdown("### Observations")
            
            # Use timeline component if available
            if 'timeline' in globals():
                events = []
                for obs in log.observations:
                    timestamp = datetime.fromisoformat(obs["timestamp"])
                    events.append({
                        "date": timestamp.strftime("%H:%M:%S"),
                        "title": "Observation",
                        "description": obs["text"],
                        "icon": "📝"
                    })
                
                timeline(events)
            else:
                # Fallback to standard display
                for i, obs in enumerate(log.observations):
                    timestamp = datetime.fromisoformat(obs["timestamp"])
                    st.markdown(f"**{timestamp.strftime('%H:%M:%S')}**")
                    st.markdown(obs["text"])
                    if i < len(log.observations) - 1:
                        st.divider()
        
        # Visualization if available
        if FRACTAL_VISUALIZER_AVAILABLE and log.cognitive_resonance is not None:
            st.markdown("### Fractal Resonance Pattern")
            
            try:
                # Create a fractal pattern based on ritual results
                params = FractalParams(
                    type="mandelbrot",
                    iterations=100,
                    resolution=400,
                    color_scale="Viridis",
                    decay_rate=0.02 * (1 - log.cognitive_resonance)  # More decay for lower CR
                )
                
                fig = fractal_visualizer.create_fractal_figure(params)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to create visualization: {str(e)}")
        
        # Notes
        if log.notes:
            st.markdown("### Notes")
            st.markdown(log.notes)
        
        # Actions
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Back to Rituals", key="back_to_rituals"):
                st.session_state.show_ritual_results = False
                st.session_state.last_completed_log = None
                st.experimental_rerun()
        
        with col2:
            if ritual and st.button("Perform Again", key="perform_again"):
                log = self.engine.start_ritual(ritual.id)
                if log:
                    st.session_state.ritual_timer_active = True
                    st.session_state.ritual_timer_start = datetime.now()
                    st.session_state.ritual_step_index = 0
                    st.session_state.show_ritual_results = False
                    st.session_state.last_completed_log = None
                    st.success(f"Started ritual: {ritual.name}")
                    st.experimental_rerun()
                else:
                    st.error("Failed to start ritual")
    
    def _render_cognitive_resonance_gauge(self, value: float):
        """Render a gauge for cognitive resonance"""
        # Use progress ring if available
        if 'progress_ring' in globals():
            progress_ring(value * 100, 100, color="#6E44FF")
            st.markdown(f"<div style='text-align: center;'>Cognitive Resonance: {value:.2f}</div>", unsafe_allow_html=True)
        else:
            # Fallback to text
            st.markdown(f"**Cognitive Resonance:** {value:.2f}")
            st.progress(value)
    
    def render_ritual_analytics(self):
        """Render the ritual analytics interface"""
        st.markdown("### 📊 Ritual Analytics")
        
        # Get analytics data
        analytics = self.engine.get_ritual_analytics()
        
        if analytics["total_performances"] == 0:
            st.info("No ritual data available for analysis")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'metric_card' in globals():
                metric_card("Rituals", str(analytics["total_rituals"]))
            else:
                st.metric("Rituals", analytics["total_rituals"])
        
        with col2:
            if 'metric_card' in globals():
                metric_card("Performances", str(analytics["total_performances"]))
            else:
                st.metric("Performances", analytics["total_performances"])
        
        with col3:
            if 'metric_card' in globals():
                metric_card("Completion Rate", f"{analytics['completion_rate']}%")
            else:
                st.metric("Completion Rate", f"{analytics['completion_rate']}%")
        
        with col4:
            if 'metric_card' in globals():
                metric_card("Avg. Effectiveness", f"{analytics['average_effectiveness']:.1f}/10")
            else:
                st.metric("Avg. Effectiveness", f"{analytics['average_effectiveness']:.1f}/10")
        
        # Ritual type distribution
        st.markdown("#### Ritual Type Distribution")
        
        ritual_types = list(analytics["ritual_counts"].keys())
        ritual_counts = list(analytics["ritual_counts"].values())
        
        if ritual_types:
            fig = px.pie(
                names=[rt.replace("_", " ").title() for rt in ritual_types],
                values=ritual_counts,
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            
            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No ritual type data available")
        
        # Time distribution
        st.markdown("#### Time of Day Distribution")
        
        time_labels = list(analytics["time_distribution"].keys())
        time_values = list(analytics["time_distribution"].values())
        
        if any(time_values):
            fig = px.bar(
                x=[t.capitalize() for t in time_labels],
                y=time_values,
                color=time_values,
                color_continuous_scale="Viridis"
            )
            
            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Time of Day",
                yaxis_title="Number of Rituals",
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time distribution data available")
        
        # Recent logs
        st.markdown("#### Recent Ritual Logs")
        
        logs = self.engine.get_logs(limit=5)
        if logs:
            for log in logs:
                start_time = datetime.fromisoformat(log.start_time)
                status_color = {
                    "completed": "#66BB6A",
                    "in_progress": "#64B5F6",
                    "interrupted": "#FFB74D"
                }.get(log.status, "#9E9E9E")
                
                log_html = f"""
                <div style="background-color: rgba(30, 40, 50, 0.8); border-radius: 10px; padding: 15px; margin-bottom: 10px; border-left: 4px solid {status_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="font-weight: 500;">{log.ritual_name}</div>
                        <div style="font-size: 0.8rem; color: #aaa;">{start_time.strftime('%Y-%m-%d %H:%M')}</div>
                    </div>
                    <div style="margin-top: 5px; font-size: 0.9rem;">
                        Status: <span style="color: {status_color};">{log.status.title()}</span>
                        {f' • Duration: {log.duration:.1f} min' if log.duration else ''}
                        {f' • CR: {log.cognitive_resonance:.2f}' if log.cognitive_resonance is not None else ''}
                    </div>
                </div>
                """
                
                st.markdown(log_html, unsafe_allow_html=True)
                
                if st.button("View Details", key=f"view_log_{log.id}"):
                    st.session_state.show_ritual_results = True
                    st.session_state.last_completed_log = log.id
                    st.experimental_rerun()
        else:
            st.info("No logs available")
    
    def render(self):
        """Render the main interface"""
        # Apply WiltonOS theme if available
        if 'apply_wiltonos_theme' in globals():
            apply_wiltonos_theme()
        
        # Title and introduction
        if 'glow_text' in globals():
            glow_text("# WiltonOS Ritual Engine")
        else:
            st.title("WiltonOS Ritual Engine")
        
        st.markdown("""
        Define, perform, and track repeatable symbolic actions with time
        and location awareness. Create rituals that integrate with the
        entire WiltonOS ecosystem.
        """)
        
        # Main tabs
        tabs = st.tabs([
            "🌟 Perform", 
            "✏️ Create & Edit",
            "📊 Analytics"
        ])
        
        # Perform tab
        with tabs[0]:
            self.render_ritual_performer()
        
        # Create & Edit tab
        with tabs[1]:
            col1, col2 = st.columns(2)
            
            with col1:
                self.render_ritual_creator()
            
            with col2:
                self.render_ritual_editor()
        
        # Analytics tab
        with tabs[2]:
            self.render_ritual_analytics()

# Create singleton instance
ritual_interface = RitualInterface()

def render_interface():
    """Render the ritual interface"""
    ritual_interface.render()

if __name__ == "__main__":
    render_interface()