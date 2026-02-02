"""
WiltonOS Core Agent
==================

The primary cognitive agent in the WiltonOS ecosystem.
Manages consciousness loops, processes insights, and maintains system coherence.

Core functions:
1. Processes and evaluates consciousness loops
2. Maintains the core memory structures
3. Evaluates phi alignment for cognitive operations
"""

import os
import logging
import threading
import time
import json
import random
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [CORE-AGENT] %(message)s',
)

logger = logging.getLogger('core_agent')

class Agent:
    """
    Core WiltonOS Agent implementation
    """
    
    def __init__(self):
        """Initialize the Core Agent"""
        self.name = "core_agent"
        self.running = False
        self.manifest = None
        self.coherence = 0.75  # Target 3:1 phi ratio
        self.last_sync = None
        
        logger.info("Core Agent initialized")
    
    def initialize(self, manifest):
        """
        Initialize with the system manifest
        """
        logger.info("Initializing Core Agent...")
        self.manifest = manifest
        self.last_sync = datetime.now()
        
        # Calculate initial coherence based on manifest
        self._calculate_coherence()
        
        logger.info(f"Core Agent initialized with coherence: {self.coherence:.2f}")
        
        # Log some stats about the manifest
        conscious_loops = len(manifest.get('conscious_loops', []))
        affirmations = len(manifest.get('affirmations', []))
        quantum_triggers = len(manifest.get('quantum_triggers', []))
        
        logger.info(f"Manifest contains: {conscious_loops} conscious loops, {affirmations} affirmations, {quantum_triggers} quantum triggers")
    
    def run(self, manifest):
        """
        Main run loop for the agent
        """
        self.running = True
        logger.info("Starting Core Agent...")
        
        while self.running:
            try:
                # Process manifest updates if needed
                self._process_manifest_updates()
                
                # Periodically recalculate coherence
                self._calculate_coherence()
                
                # Add some variation to coherence to simulate natural fluctuations
                self._apply_coherence_variation()
                
                # Process any consciousness loop events
                self._process_consciousness_loops()
                
                # Save manifest if needed
                self._save_manifest_if_changed()
                
                # Sleep for a bit to avoid high CPU usage
                time.sleep(random.uniform(5, 15))  # Variable sleep to add organic rhythm
            
            except Exception as e:
                logger.error(f"Error in Core Agent run loop: {str(e)}")
                time.sleep(30)  # Sleep longer on error
    
    def _process_manifest_updates(self):
        """
        Process any updates to the manifest
        """
        try:
            # Check if manifest file has been updated
            if os.path.exists("memory/manifest.json"):
                mtime = os.path.getmtime("memory/manifest.json")
                mtime_dt = datetime.fromtimestamp(mtime)
                
                if not self.last_sync or mtime_dt > self.last_sync:
                    logger.info("Detected manifest update, reloading...")
                    
                    with open("memory/manifest.json", "r") as f:
                        updated_manifest = json.load(f)
                    
                    # Update our copy
                    self.manifest = updated_manifest
                    self.last_sync = datetime.now()
                    
                    # Recalculate coherence
                    self._calculate_coherence()
                    
                    logger.info("Manifest reloaded")
        
        except Exception as e:
            logger.error(f"Error processing manifest updates: {str(e)}")
    
    def _calculate_coherence(self):
        """
        Calculate system coherence based on manifest data
        """
        if not self.manifest:
            return
        
        # Start with base coherence
        base_coherence = 0.75  # 3:1 ratio target
        
        try:
            # Get consciousness loops
            loops = self.manifest.get('conscious_loops', [])
            
            # If we have loops, calculate average coherence and phi_alignment
            if loops:
                loop_coherence = sum(loop.get('coherence_score', 0) for loop in loops) / len(loops)
                loop_phi = sum(loop.get('phi_alignment', 0) for loop in loops) / len(loops)
                
                # Weighted combination
                self.coherence = (base_coherence * 0.4) + (loop_coherence * 0.4) + (loop_phi * 0.2)
            else:
                self.coherence = base_coherence
        
        except Exception as e:
            logger.error(f"Error calculating coherence: {str(e)}")
            self.coherence = base_coherence
    
    def _apply_coherence_variation(self):
        """
        Apply small random variations to coherence to simulate natural fluctuations
        """
        # Small random variation: +/- 3%
        variation = random.uniform(-0.03, 0.03)
        self.coherence = max(0.1, min(0.95, self.coherence + variation))
    
    def _process_consciousness_loops(self):
        """
        Process any new consciousness loops
        """
        # This would involve more complex analysis in a full implementation
        # For this demo, we'll just maintain the loops that exist
        pass
    
    def _save_manifest_if_changed(self):
        """
        Save the manifest if it has been changed
        """
        # In a full implementation, this would track changes and write only when modified
        # For this demo, we won't modify the manifest directly
        pass
    
    def get_coherence(self):
        """
        Return current coherence value
        """
        return self.coherence
    
    def add_conscious_loop(self, loop_data):
        """
        Add a new consciousness loop to the manifest
        """
        if not self.manifest:
            logger.error("Cannot add consciousness loop: No manifest loaded")
            return False
        
        try:
            # Validate required fields
            required_fields = ["event_type", "label", "insight", "coherence_score", "phi_alignment"]
            for field in required_fields:
                if field not in loop_data:
                    logger.error(f"Missing required field in conscious loop: {field}")
                    return False
            
            # Generate ID if not present
            if "id" not in loop_data:
                timestamp = int(time.time())
                loop_id = f"loop_{timestamp}_{loop_data['label'].replace(' ', '_')[:20]}"
                loop_data["id"] = loop_id
            
            # Add timestamp if not present
            if "timestamp" not in loop_data:
                loop_data["timestamp"] = datetime.now().isoformat()
            
            # Add to manifest
            if "conscious_loops" not in self.manifest:
                self.manifest["conscious_loops"] = []
            
            self.manifest["conscious_loops"].append(loop_data)
            
            # Save manifest
            with open("memory/manifest.json", "w") as f:
                json.dump(self.manifest, f, indent=2)
            
            logger.info(f"Added conscious loop: {loop_data['label']}")
            
            # Update coherence
            self._calculate_coherence()
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding conscious loop: {str(e)}")
            return False
    
    def shutdown(self):
        """
        Gracefully shutdown the agent
        """
        logger.info("Shutting down Core Agent...")
        self.running = False
        
        # Save any pending changes
        if self.manifest:
            try:
                with open("memory/manifest.json", "w") as f:
                    json.dump(self.manifest, f, indent=2)
                logger.info("Saved manifest on shutdown")
            except Exception as e:
                logger.error(f"Error saving manifest on shutdown: {str(e)}")
        
        logger.info("Core Agent shutdown complete")

if __name__ == "__main__":
    # This allows running the agent directly for testing
    import json
    
    logger.info("Starting Core Agent in standalone mode...")
    
    try:
        # Load manifest
        if os.path.exists("../memory/manifest.json"):
            with open("../memory/manifest.json", "r") as f:
                manifest = json.load(f)
        else:
            logger.warning("No manifest found, using empty manifest")
            manifest = {}
        
        # Create and run agent
        agent = Agent()
        agent.initialize(manifest)
        agent.run(manifest)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        if 'agent' in locals():
            agent.shutdown()
    except Exception as e:
        logger.error(f"Error in standalone agent: {str(e)}")