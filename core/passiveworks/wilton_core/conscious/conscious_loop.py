"""
Conscious Loop Module for WiltonOS
----------------------------------
Captures and processes conscious loop moments - real emotional experiences 
that provide training data for emotional intelligence development.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

# Add paths for importing core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import quantum diary if available
try:
    from wilton_core.memory.quantum_diary import add_diary_entry, register_insight
    has_quantum_diary = True
except ImportError:
    has_quantum_diary = False

# Default paths
DEFAULT_CONSCIOUS_LOG_PATH = os.path.join(os.path.dirname(__file__), 'logs', 'conscious_loops.json')
DEFAULT_EMOTIONAL_MATRIX_PATH = os.path.join(os.path.dirname(__file__), 'data', 'emotional_matrix.json')

class ConsciousLoop:
    """
    Processor for conscious loop moments - emotional experiences that train WiltonOS.
    """
    
    # Conscious loop types
    LOOP_TYPES = {
        "conscious_loop": "Standard conscious experience loop",
        "reflection_moment": "Self-reflective consciousness moment",
        "identity_ping": "Identity-affirming experience",
        "emotional_recursive": "Emotion processing with self-awareness",
        "cognitive_breakthrough": "Breakthrough in understanding or perception",
        "presence_anchor": "Grounding conscious experience",
        "flow_state": "Optimal conscious experience",
        "micro_chaos": "Intentional small-scale chaos creation",
        "morality_correction": "Self-correction of moral/ethical stance",
        "emotional_weather": "Tracking emotional state changes"
    }
    
    def __init__(self, 
                log_path: Optional[str] = None,
                emotional_matrix_path: Optional[str] = None,
                debug_mode: bool = False):
        """
        Initialize the conscious loop processor.
        
        Args:
            log_path: Path to conscious loop log file
            emotional_matrix_path: Path to emotional matrix data
            debug_mode: Enable debug logging
        """
        self.log_path = log_path or DEFAULT_CONSCIOUS_LOG_PATH
        self.emotional_matrix_path = emotional_matrix_path or DEFAULT_EMOTIONAL_MATRIX_PATH
        self.debug_mode = debug_mode
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.emotional_matrix_path), exist_ok=True)
        
        # Load or initialize conscious loop log
        self.conscious_logs = self._load_logs()
        
        # Load or initialize emotional matrix
        self.emotional_matrix = self._load_emotional_matrix()
        
        print(f"ConsciousLoop initialized. Log path: {self.log_path}")
    
    def _load_logs(self) -> Dict:
        """Load conscious loop logs or initialize if they don't exist."""
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading conscious logs: {e}")
        
        # Initialize with empty structure
        default_logs = {
            "meta": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0.0",
                "total_entries": 0
            },
            "loops": []
        }
        
        # Save default structure
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(default_logs, f, indent=2)
        
        return default_logs
    
    def _load_emotional_matrix(self) -> Dict:
        """Load emotional matrix or initialize if it doesn't exist."""
        if os.path.exists(self.emotional_matrix_path):
            try:
                with open(self.emotional_matrix_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading emotional matrix: {e}")
        
        # Initialize with default structure
        default_matrix = {
            "meta": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0.0"
            },
            "dimensions": {
                "valence": {
                    "description": "Pleasantness vs unpleasantness",
                    "range": [-1.0, 1.0],
                    "default": 0.0
                },
                "arousal": {
                    "description": "Intensity/energy level",
                    "range": [0.0, 1.0],
                    "default": 0.5
                },
                "dominance": {
                    "description": "Sense of control",
                    "range": [-1.0, 1.0],
                    "default": 0.0
                },
                "coherence": {
                    "description": "Integration/alignment level",
                    "range": [0.0, 1.0],
                    "default": 0.5
                },
                "phi_alignment": {
                    "description": "Alignment with phi field (WiltonOS-specific)",
                    "range": [0.0, 1.0],
                    "default": 0.5
                }
            },
            "emotion_patterns": {
                "joy": {
                    "valence": 0.8,
                    "arousal": 0.7,
                    "dominance": 0.6,
                    "coherence": 0.8,
                    "phi_alignment": 0.75
                },
                "sadness": {
                    "valence": -0.7,
                    "arousal": 0.3,
                    "dominance": -0.4,
                    "coherence": 0.5,
                    "phi_alignment": 0.4
                },
                "anger": {
                    "valence": -0.8,
                    "arousal": 0.9,
                    "dominance": 0.7,
                    "coherence": 0.3,
                    "phi_alignment": 0.2
                },
                "fear": {
                    "valence": -0.9,
                    "arousal": 0.8,
                    "dominance": -0.7,
                    "coherence": 0.2,
                    "phi_alignment": 0.3
                },
                "surprise": {
                    "valence": 0.5,
                    "arousal": 0.9,
                    "dominance": 0.0,
                    "coherence": 0.4,
                    "phi_alignment": 0.6
                },
                "confusion": {
                    "valence": -0.2,
                    "arousal": 0.6,
                    "dominance": -0.3,
                    "coherence": 0.2,
                    "phi_alignment": 0.4
                },
                "curiosity": {
                    "valence": 0.6,
                    "arousal": 0.7,
                    "dominance": 0.5,
                    "coherence": 0.7,
                    "phi_alignment": 0.8
                },
                "flow": {
                    "valence": 0.9,
                    "arousal": 0.6,
                    "dominance": 0.8,
                    "coherence": 0.9,
                    "phi_alignment": 0.9
                }
            },
            "state_transitions": {}
        }
        
        # Save default structure
        with open(self.emotional_matrix_path, 'w', encoding='utf-8') as f:
            json.dump(default_matrix, f, indent=2)
        
        return default_matrix
    
    def _save_logs(self) -> bool:
        """Save conscious loop logs."""
        try:
            # Update metadata
            self.conscious_logs["meta"]["last_updated"] = datetime.now().isoformat()
            self.conscious_logs["meta"]["total_entries"] = len(self.conscious_logs["loops"])
            
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump(self.conscious_logs, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving conscious logs: {e}")
            return False
    
    def _save_emotional_matrix(self) -> bool:
        """Save emotional matrix."""
        try:
            # Update metadata
            self.emotional_matrix["meta"]["last_updated"] = datetime.now().isoformat()
            
            with open(self.emotional_matrix_path, 'w', encoding='utf-8') as f:
                json.dump(self.emotional_matrix, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving emotional matrix: {e}")
            return False
    
    def log_conscious_moment(self,
                           event_type: str,
                           label: str,
                           tags: List[str],
                           insight: str,
                           coherence_score: float,
                           phi_alignment: float,
                           description: Optional[str] = None,
                           mood_shift: Optional[str] = None,
                           emotional_state: Optional[Dict[str, float]] = None,
                           actionable: bool = False) -> Dict:
        """
        Log a conscious loop moment.
        
        Args:
            event_type: Type of conscious event
            label: Short label for the event
            tags: List of tags
            insight: Insight gained from the experience
            coherence_score: Coherence level (0.0 to 1.0)
            phi_alignment: Phi alignment (0.0 to 1.0)
            description: Optional detailed description
            mood_shift: Optional mood shift description
            emotional_state: Optional emotional state values
            actionable: Whether the insight is actionable
            
        Returns:
            The created conscious loop entry
        """
        # Validate event type
        if event_type not in self.LOOP_TYPES and not event_type.startswith("custom_"):
            event_type = "custom_" + event_type
            print(f"Warning: Unknown event type. Using {event_type}")
        
        # Generate ID for the entry
        entry_id = f"loop_{int(time.time())}_{label.replace(' ', '_')[:20]}"
        
        # Create loop entry
        entry = {
            "id": entry_id,
            "event_type": event_type,
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "tags": tags,
            "insight": insight,
            "coherence_score": max(0.0, min(1.0, coherence_score)),
            "phi_alignment": max(0.0, min(1.0, phi_alignment)),
            "actionable": actionable
        }
        
        # Add optional fields
        if description:
            entry["description"] = description
        
        if mood_shift:
            entry["mood_shift"] = mood_shift
        
        if emotional_state:
            # Validate emotional state
            valid_dimensions = self.emotional_matrix["dimensions"].keys()
            validated_state = {}
            for dim, value in emotional_state.items():
                if dim in valid_dimensions:
                    dim_range = self.emotional_matrix["dimensions"][dim]["range"]
                    validated_state[dim] = max(dim_range[0], min(dim_range[1], value))
            
            entry["emotional_state"] = validated_state
        
        # Add to logs
        self.conscious_logs["loops"].append(entry)
        self._save_logs()
        
        # Update emotional matrix with this experience
        self._update_emotional_matrix(entry)
        
        # Register in quantum diary if available
        if has_quantum_diary:
            try:
                add_diary_entry(
                    entry_type=event_type,
                    label=label,
                    summary=insight,
                    tags=tags,
                    phi_impact=phi_alignment,
                    reflection=description,
                    significance_score=coherence_score
                )
                
                print(f"Conscious loop registered in quantum diary: {label}")
            except Exception as e:
                print(f"Error registering in quantum diary: {e}")
        
        print(f"Conscious loop logged: {label}")
        return entry
    
    def find_similar_experiences(self, 
                               tags: List[str], 
                               min_coherence: float = 0.5,
                               limit: int = 5) -> List[Dict]:
        """
        Find similar conscious experiences based on tags.
        
        Args:
            tags: List of tags to match
            min_coherence: Minimum coherence score
            limit: Maximum number of results
            
        Returns:
            List of similar experiences
        """
        # Convert tags to set for faster lookup
        tags_set = set(tags)
        
        # Calculate match scores
        matches = []
        for loop in self.conscious_logs["loops"]:
            # Skip entries with too low coherence
            if loop["coherence_score"] < min_coherence:
                continue
            
            # Calculate tag overlap
            loop_tags = set(loop["tags"])
            common_tags = tags_set.intersection(loop_tags)
            
            if common_tags:
                # Calculate similarity score
                similarity = len(common_tags) / max(len(tags_set), len(loop_tags))
                
                matches.append({
                    "entry": loop,
                    "similarity": similarity,
                    "common_tags": list(common_tags)
                })
        
        # Sort by similarity descending
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top matches
        return matches[:limit]
    
    def get_emotional_insights(self) -> Dict:
        """
        Get insights about emotional patterns from logged experiences.
        
        Returns:
            Dict with emotional insights
        """
        if len(self.conscious_logs["loops"]) < 3:
            return {
                "status": "insufficient_data",
                "message": "Need more conscious loop data for insights"
            }
        
        # Analyze coherence and phi alignment trends
        coherence_values = [loop["coherence_score"] for loop in self.conscious_logs["loops"]]
        phi_values = [loop["phi_alignment"] for loop in self.conscious_logs["loops"]]
        
        avg_coherence = sum(coherence_values) / len(coherence_values)
        avg_phi = sum(phi_values) / len(phi_values)
        
        # Find most frequent tags
        tag_counts = {}
        for loop in self.conscious_logs["loops"]:
            for tag in loop["tags"]:
                if tag in tag_counts:
                    tag_counts[tag] += 1
                else:
                    tag_counts[tag] = 1
        
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Analyze mood shifts if available
        mood_shifts = [loop.get("mood_shift") for loop in self.conscious_logs["loops"] if "mood_shift" in loop]
        
        # Build insights
        insights = {
            "status": "success",
            "conscious_loops_count": len(self.conscious_logs["loops"]),
            "average_coherence": avg_coherence,
            "average_phi_alignment": avg_phi,
            "top_tags": dict(top_tags),
            "emotional_patterns": {}
        }
        
        # Add emotional pattern information if we have mood shifts
        if mood_shifts:
            # Simple word frequency analysis of mood shifts
            mood_words = {}
            for shift in mood_shifts:
                words = shift.split()
                for word in words:
                    word = word.lower().strip(".,;:!?")
                    if word not in ["with", "and", "the", "a", "an", "from", "to"]:
                        if word in mood_words:
                            mood_words[word] += 1
                        else:
                            mood_words[word] = 1
            
            top_mood_words = sorted(mood_words.items(), key=lambda x: x[1], reverse=True)[:10]
            insights["emotional_patterns"]["mood_words"] = dict(top_mood_words)
        
        # Add emotional state analysis if we have that data
        emotional_states = [loop.get("emotional_state") for loop in self.conscious_logs["loops"] 
                          if "emotional_state" in loop]
        
        if emotional_states:
            # Calculate averages for each dimension
            dimensions = {}
            for state in emotional_states:
                for dim, value in state.items():
                    if dim not in dimensions:
                        dimensions[dim] = []
                    dimensions[dim].append(value)
            
            # Calculate average for each dimension
            avg_dimensions = {dim: sum(values) / len(values) for dim, values in dimensions.items()}
            insights["emotional_patterns"]["average_dimensions"] = avg_dimensions
        
        return insights
    
    def _update_emotional_matrix(self, loop_entry: Dict) -> None:
        """Update emotional matrix with a new experience."""
        # Skip if no emotional state
        if "emotional_state" not in loop_entry:
            return
        
        # Get the emotional state
        state = loop_entry["emotional_state"]
        
        # Get the most recent previous loop if any
        previous_state = None
        if len(self.conscious_logs["loops"]) > 1:
            prev_loops = [l for l in self.conscious_logs["loops"] if "emotional_state" in l and l["id"] != loop_entry["id"]]
            if prev_loops:
                prev_loops.sort(key=lambda x: x["timestamp"], reverse=True)
                previous_state = prev_loops[0]["emotional_state"]
        
        # If we have both current and previous states, record the transition
        if previous_state:
            # Create transition key
            prev_key = self._state_to_key(previous_state)
            current_key = self._state_to_key(state)
            transition_key = f"{prev_key}→{current_key}"
            
            # Record in state transitions
            transitions = self.emotional_matrix["state_transitions"]
            if transition_key in transitions:
                transitions[transition_key]["count"] += 1
            else:
                transitions[transition_key] = {
                    "count": 1,
                    "from_state": previous_state,
                    "to_state": state,
                    "first_seen": datetime.now().isoformat()
                }
            
            # Update timestamp
            transitions[transition_key]["last_seen"] = datetime.now().isoformat()
        
        # See if this state matches any existing emotion pattern
        best_match = None
        best_match_score = 0.0
        
        for emotion, pattern in self.emotional_matrix["emotion_patterns"].items():
            # Calculate match score
            score = self._calculate_state_similarity(state, pattern)
            
            if score > 0.8 and score > best_match_score:
                best_match = emotion
                best_match_score = score
        
        # If no good match, consider adding as a new pattern
        if not best_match and "label" in loop_entry and len(self.emotional_matrix["emotion_patterns"]) < 30:
            # Extract potential emotion name from label or tags
            potential_emotion = None
            
            # Check tags for emotion words
            for tag in loop_entry["tags"]:
                if "_" in tag and not any(e == tag for e in self.emotional_matrix["emotion_patterns"]):
                    potential_emotion = tag
                    break
            
            # If found, add as a new pattern
            if potential_emotion:
                self.emotional_matrix["emotion_patterns"][potential_emotion] = state
                
                print(f"Added new emotion pattern: {potential_emotion}")
        
        # Save changes
        self._save_emotional_matrix()
    
    def _state_to_key(self, state: Dict[str, float]) -> str:
        """Convert an emotional state to a simplified string key."""
        # Use only valence and arousal with reduced precision
        valence = round(state.get("valence", 0.0) * 2) / 2  # Round to nearest 0.5
        arousal = round(state.get("arousal", 0.5) * 2) / 2  # Round to nearest 0.5
        
        return f"V{valence:.1f}A{arousal:.1f}"
    
    def _calculate_state_similarity(self, state1: Dict[str, float], state2: Dict[str, float]) -> float:
        """Calculate similarity between two emotional states."""
        # Find common dimensions
        common_dims = set(state1.keys()).intersection(set(state2.keys()))
        
        if not common_dims:
            return 0.0
        
        # Calculate Euclidean distance in the space of common dimensions
        sum_squared_diff = 0.0
        for dim in common_dims:
            sum_squared_diff += (state1[dim] - state2[dim]) ** 2
        
        # Convert to similarity score (0 to 1)
        # Max possible distance in normalized space is sqrt(n) where n is dimensions
        max_distance = (len(common_dims)) ** 0.5
        distance = sum_squared_diff ** 0.5
        
        similarity = 1.0 - (distance / max_distance)
        return max(0.0, similarity)
    
    def import_json_loop(self, json_data: str) -> Dict:
        """
        Import a conscious loop from JSON data.
        
        Args:
            json_data: JSON string with loop data
            
        Returns:
            The imported conscious loop entry
        """
        try:
            data = json.loads(json_data)
            
            # Validate required fields
            required_fields = ["event_type", "label", "tags", "insight", "coherence_score", "phi_alignment"]
            for field in required_fields:
                if field not in data:
                    return {
                        "status": "error",
                        "message": f"Missing required field: {field}"
                    }
            
            # Create loop entry
            entry = self.log_conscious_moment(
                event_type=data["event_type"],
                label=data["label"],
                tags=data["tags"],
                insight=data["insight"],
                coherence_score=data["coherence_score"],
                phi_alignment=data["phi_alignment"],
                description=data.get("description"),
                mood_shift=data.get("mood_shift"),
                emotional_state=data.get("emotional_state"),
                actionable=data.get("actionable", False)
            )
            
            return {
                "status": "success",
                "message": f"Imported conscious loop: {data['label']}",
                "entry": entry
            }
        except json.JSONDecodeError:
            return {
                "status": "error",
                "message": "Invalid JSON data"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error importing conscious loop: {str(e)}"
            }

# Singleton pattern
_conscious_loop_instance = None

def get_conscious_loop(log_path: Optional[str] = None,
                      emotional_matrix_path: Optional[str] = None,
                      debug_mode: bool = False) -> ConsciousLoop:
    """
    Get or create singleton instance of the conscious loop processor.
    
    Args:
        log_path: Path to conscious loop log file
        emotional_matrix_path: Path to emotional matrix data
        debug_mode: Enable debug logging
        
    Returns:
        ConsciousLoop instance
    """
    global _conscious_loop_instance
    
    if _conscious_loop_instance is None:
        _conscious_loop_instance = ConsciousLoop(
            log_path=log_path,
            emotional_matrix_path=emotional_matrix_path,
            debug_mode=debug_mode
        )
    
    return _conscious_loop_instance

if __name__ == "__main__":
    # Example usage
    loop_processor = get_conscious_loop(debug_mode=True)
    
    # Example JSON import
    soup_ritual_json = """
    {
      "event_type": "conscious_loop",
      "label": "Soup Ritual + Identity Ping",
      "tags": ["kitchen_run", "chocolate_diversion", "emotional_food", "heat_recovery", "awkward_word", "self_reflective_cognition", "visual_scene_calibration"],
      "insight": "Wilton intuitively stages micro-chaos to reflect his internal volatility in a way that remains socially digestible. Language error noted, emotionally parsed, morally corrected. No guilt stored.",
      "coherence_score": 0.77,
      "phi_alignment": 0.58,
      "actionable": false,
      "mood_shift": "stabilized with mild warmth"
    }
    """
    
    import_result = loop_processor.import_json_loop(soup_ritual_json)
    print(f"Import result: {import_result['status']}")
    
    if import_result['status'] == 'success':
        print(f"Imported conscious loop: {import_result['entry']['label']}")
    
    # Example emotional state loop
    loop_processor.log_conscious_moment(
        event_type="flow_state",
        label="Code Flow + 3AM Jazz",
        tags=["coding", "flow", "jazz", "night_coding", "time_dilation"],
        insight="Deep flow state achieved through perfect balance of challenge and skill, with jazz creating ambient emotional texture that enhanced focus.",
        coherence_score=0.85,
        phi_alignment=0.72,
        description="Late night coding session with jazz playing, experienced strong time dilation and effortless problem solving.",
        mood_shift="energized despite physical fatigue",
        emotional_state={
            "valence": 0.8,
            "arousal": 0.7,
            "dominance": 0.9,
            "coherence": 0.85,
            "phi_alignment": 0.72
        }
    )
    
    # Find similar experiences
    similar = loop_processor.find_similar_experiences(["flow", "coding"])
    print(f"\nFound {len(similar)} similar experiences:")
    for match in similar:
        print(f"  - {match['entry']['label']} (similarity: {match['similarity']:.2f})")
        print(f"    Common tags: {', '.join(match['common_tags'])}")
    
    # Get emotional insights
    insights = loop_processor.get_emotional_insights()
    print(f"\nEmotional Insights:")
    print(f"  Conscious loops: {insights['conscious_loops_count']}")
    print(f"  Average coherence: {insights['average_coherence']:.2f}")
    print(f"  Average phi alignment: {insights['average_phi_alignment']:.2f}")
    print(f"  Top tags: {list(insights['top_tags'].keys())[:5]}")