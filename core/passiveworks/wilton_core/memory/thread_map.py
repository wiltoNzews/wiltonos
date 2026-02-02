"""
Thread Map Module for WiltonOS
------------------------------
Maps, tracks, and analyzes sequences of social media posts, messages, and interactions.
Creates a coherent narrative map with resonance scores and emergent themes.
"""

import os
import json
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

# Try to import the quantum diary for integration
try:
    from wilton_core.memory.quantum_diary import add_diary_entry
    has_diary = True
except ImportError:
    has_diary = False

# Constants
THREAD_MAP_PATH = os.path.join(os.path.dirname(__file__), 'thread_map.json')
RESONANCE_THRESHOLD = 0.65  # Minimum resonance score to be considered significant
THEME_SIMILARITY_THRESHOLD = 0.7  # Threshold for considering themes similar

class ThreadMap:
    """
    Maps social media threads and analyzes their quantum resonance.
    """
    
    def __init__(self, map_path: Optional[str] = None):
        """
        Initialize the thread map.
        
        Args:
            map_path: Optional path to the thread map JSON file
        """
        self.map_path = map_path or THREAD_MAP_PATH
        self.threads = self._load_map()
        
    def _load_map(self) -> Dict:
        """Load the thread map from disk."""
        if not os.path.exists(self.map_path):
            # Initialize with empty structure if file doesn't exist
            return {
                "threads": {},
                "meta": {
                    "last_updated": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "total_threads": 0,
                    "active_threads": 0,
                    "completed_threads": 0
                }
            }
        
        try:
            with open(self.map_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading thread map: {e}")
            # Return empty structure on error
            return {"threads": {}, "meta": {}}
    
    def _save_map(self) -> bool:
        """Save the thread map to disk."""
        try:
            with open(self.map_path, 'w', encoding='utf-8') as f:
                json.dump(self.threads, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving thread map: {e}")
            return False
    
    def create_thread(
        self, 
        title: str, 
        platform: str,
        source_url: Optional[str] = None,
        description: Optional[str] = None,
        initial_theme: Optional[str] = None,
        tags: Optional[List[str]] = None,
        related_threads: Optional[List[str]] = None,
        phi_impact: float = 0.0
    ) -> str:
        """
        Create a new thread in the map.
        
        Args:
            title: The title of the thread
            platform: Social media platform (e.g., 'twitter', 'instagram')
            source_url: URL to the original thread
            description: Optional description of the thread
            initial_theme: Initial theme of the thread
            tags: List of tags for categorization
            related_threads: List of related thread IDs
            phi_impact: Initial phi impact of this thread
            
        Returns:
            thread_id: The ID of the newly created thread
        """
        thread_id = f"thread_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()
        
        thread_data = {
            "thread_id": thread_id,
            "title": title,
            "platform": platform,
            "created_at": timestamp,
            "updated_at": timestamp,
            "source_url": source_url,
            "description": description,
            "theme_emergent": initial_theme or "undetermined",
            "theme_history": [{"timestamp": timestamp, "theme": initial_theme or "undetermined"}],
            "resonance_score": 0.0,
            "interaction_count": 0,
            "posts": [],
            "status": "active",
            "tags": tags or [],
            "related_threads": related_threads or [],
            "phi_impact": phi_impact,
            "metrics": {
                "engagement_rate": 0.0,
                "sentiment_average": 0.0,
                "virality_score": 0.0
            }
        }
        
        # Add to threads dictionary
        self.threads["threads"][thread_id] = thread_data
        
        # Update metadata
        self.threads["meta"]["last_updated"] = timestamp
        self.threads["meta"]["total_threads"] += 1
        self.threads["meta"]["active_threads"] += 1
        
        # Save the updated map
        self._save_map()
        
        # Register in quantum diary if available
        if has_diary:
            try:
                add_diary_entry(
                    entry_type="thread_impact",
                    label=f"Thread Created: {title}",
                    summary=f"New social thread tracking initiated on {platform}.",
                    reflection=description,
                    tags=["thread", platform] + (tags or []),
                    phi_impact=phi_impact,
                    thread_id=thread_id,
                    source={
                        "platform": platform,
                        "url": source_url
                    },
                    significance_score=0.2  # Initial low significance until resonance grows
                )
            except Exception as e:
                print(f"Error creating thread diary entry: {e}")
        
        return thread_id
    
    def add_post(
        self, 
        thread_id: str, 
        content: str,
        author: Optional[str] = None,
        timestamp: Optional[str] = None,
        post_url: Optional[str] = None,
        engagement: Optional[Dict[str, int]] = None,
        sentiment: Optional[float] = None,
        phi_impact: Optional[float] = None,
        tags: Optional[List[str]] = None,
        resonance_delta: Optional[float] = None
    ) -> Dict:
        """
        Add a post to an existing thread.
        
        Args:
            thread_id: ID of the thread
            content: Content of the post
            author: Author of the post
            timestamp: Timestamp of the post (ISO format)
            post_url: URL to the original post
            engagement: Dict with engagement metrics (likes, shares, etc.)
            sentiment: Sentiment score of the post (-1.0 to 1.0)
            phi_impact: Phi impact of this post
            tags: List of tags for the post
            resonance_delta: Change in resonance due to this post
            
        Returns:
            post_data: The data of the newly added post
        """
        if thread_id not in self.threads["threads"]:
            raise ValueError(f"Thread with ID {thread_id} not found")
        
        thread = self.threads["threads"][thread_id]
        current_time = datetime.now().isoformat()
        
        post_id = f"post_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        
        # Create post data
        post_data = {
            "post_id": post_id,
            "thread_id": thread_id,
            "content": content,
            "author": author,
            "timestamp": timestamp or current_time,
            "post_url": post_url,
            "engagement": engagement or {"views": 0, "likes": 0, "shares": 0, "comments": 0},
            "sentiment": sentiment or 0.0,
            "phi_impact": phi_impact or 0.0,
            "tags": tags or [],
            "resonance_delta": resonance_delta or 0.0,
            "indexed_at": current_time
        }
        
        # Add post to thread
        thread["posts"].append(post_data)
        
        # Update thread metrics
        thread["interaction_count"] += 1
        thread["updated_at"] = current_time
        
        # Update resonance score
        if resonance_delta is not None:
            thread["resonance_score"] = max(0.0, min(1.0, thread["resonance_score"] + resonance_delta))
        
        # Update engagement metrics if available
        if engagement:
            total_posts = len(thread["posts"])
            thread["metrics"]["engagement_rate"] = self._calculate_engagement_rate(thread)
            
        # Update sentiment if available
        if sentiment is not None:
            sentiments = [p.get("sentiment", 0) for p in thread["posts"] if "sentiment" in p]
            if sentiments:
                thread["metrics"]["sentiment_average"] = sum(sentiments) / len(sentiments)
        
        # Update phi impact if available
        if phi_impact is not None:
            thread["phi_impact"] += phi_impact
        
        # Save the updated map
        self._save_map()
        
        # Register significant post in quantum diary if available
        if has_diary and (phi_impact or 0) > 0.05:
            try:
                add_diary_entry(
                    entry_type="thread_impact",
                    label=f"Significant Post in Thread: {thread['title']}",
                    summary=content[:200] + ("..." if len(content) > 200 else ""),
                    reflection=f"Author: {author or 'Unknown'}, Resonance Delta: {resonance_delta or 0}",
                    tags=["thread", "post", thread["platform"]] + (tags or []),
                    phi_impact=phi_impact or 0,
                    thread_id=thread_id,
                    source={
                        "platform": thread["platform"],
                        "url": post_url or thread.get("source_url")
                    },
                    significance_score=resonance_delta or phi_impact or 0.1
                )
            except Exception as e:
                print(f"Error creating post diary entry: {e}")
        
        return post_data
    
    def update_thread_theme(self, thread_id: str, new_theme: str, reason: Optional[str] = None) -> bool:
        """
        Update the emergent theme of a thread.
        
        Args:
            thread_id: ID of the thread
            new_theme: New emergent theme
            reason: Reason for the theme update
            
        Returns:
            bool: Success status
        """
        if thread_id not in self.threads["threads"]:
            return False
        
        thread = self.threads["threads"][thread_id]
        timestamp = datetime.now().isoformat()
        
        # Record theme history
        thread["theme_history"].append({
            "timestamp": timestamp,
            "theme": new_theme,
            "reason": reason
        })
        
        # Update current theme
        thread["theme_emergent"] = new_theme
        thread["updated_at"] = timestamp
        
        # Save the updated map
        return self._save_map()
    
    def update_thread_status(self, thread_id: str, status: str) -> bool:
        """
        Update the status of a thread.
        
        Args:
            thread_id: ID of the thread
            status: New status ('active', 'completed', 'archived')
            
        Returns:
            bool: Success status
        """
        if thread_id not in self.threads["threads"]:
            return False
        
        thread = self.threads["threads"][thread_id]
        timestamp = datetime.now().isoformat()
        
        # Update metadata counters
        if thread["status"] == "active" and status != "active":
            self.threads["meta"]["active_threads"] -= 1
            
        if status == "completed" and thread["status"] != "completed":
            self.threads["meta"]["completed_threads"] += 1
        elif thread["status"] == "completed" and status != "completed":
            self.threads["meta"]["completed_threads"] -= 1
            
        if status == "active" and thread["status"] != "active":
            self.threads["meta"]["active_threads"] += 1
        
        # Update thread status
        thread["status"] = status
        thread["updated_at"] = timestamp
        
        # Save the updated map
        return self._save_map()
    
    def get_thread(self, thread_id: str) -> Optional[Dict]:
        """Get a thread by ID."""
        return self.threads["threads"].get(thread_id)
    
    def get_threads_by_tag(self, tag: str) -> List[Dict]:
        """Get threads with a specific tag."""
        return [thread for thread in self.threads["threads"].values() 
                if tag in thread.get("tags", [])]
    
    def get_threads_by_platform(self, platform: str) -> List[Dict]:
        """Get threads from a specific platform."""
        return [thread for thread in self.threads["threads"].values() 
                if thread.get("platform") == platform]
    
    def get_high_resonance_threads(self, threshold: Optional[float] = None) -> List[Dict]:
        """Get threads with resonance above the threshold."""
        threshold = threshold or RESONANCE_THRESHOLD
        return [thread for thread in self.threads["threads"].values() 
                if thread.get("resonance_score", 0) >= threshold]
    
    def get_thread_metrics(self, thread_id: str) -> Dict:
        """Get detailed metrics for a thread."""
        thread = self.get_thread(thread_id)
        if not thread:
            return {}
        
        # Calculate current metrics
        return {
            "interaction_count": thread["interaction_count"],
            "resonance_score": thread["resonance_score"],
            "phi_impact": thread["phi_impact"],
            "engagement_rate": thread["metrics"]["engagement_rate"],
            "sentiment_average": thread["metrics"]["sentiment_average"],
            "virality_score": thread["metrics"]["virality_score"],
            "post_count": len(thread["posts"]),
            "theme": thread["theme_emergent"],
            "theme_stability": self._calculate_theme_stability(thread),
            "last_activity": thread["updated_at"],
            "status": thread["status"],
            "age_days": self._calculate_age_days(thread)
        }
    
    def find_related_threads(self, thread_id: str, by_theme: bool = True) -> List[Dict]:
        """
        Find threads related to the given thread.
        
        Args:
            thread_id: ID of the thread
            by_theme: Whether to find relations by theme similarity
            
        Returns:
            List of related threads
        """
        thread = self.get_thread(thread_id)
        if not thread:
            return []
        
        related = []
        
        # Include explicitly related threads
        for related_id in thread.get("related_threads", []):
            related_thread = self.get_thread(related_id)
            if related_thread:
                related.append(related_thread)
        
        # Find threads with similar themes if requested
        if by_theme:
            current_theme = thread["theme_emergent"]
            for tid, t in self.threads["threads"].items():
                if tid != thread_id and tid not in thread.get("related_threads", []):
                    # Simple string similarity for themes
                    # In a real implementation, this could use semantic similarity
                    if self._theme_similarity(current_theme, t["theme_emergent"]) >= THEME_SIMILARITY_THRESHOLD:
                        related.append(t)
        
        return related
    
    def _calculate_engagement_rate(self, thread: Dict) -> float:
        """Calculate engagement rate for a thread."""
        total_engagement = 0
        for post in thread["posts"]:
            eng = post.get("engagement", {})
            # Basic formula: sum of all engagement types
            post_engagement = (
                eng.get("likes", 0) + 
                eng.get("shares", 0) * 3 +  # Shares weighted more heavily
                eng.get("comments", 0) * 2  # Comments weighted more heavily
            )
            total_engagement += post_engagement
        
        # Number of posts with at least some engagement
        active_posts = sum(1 for p in thread["posts"] if sum(p.get("engagement", {}).values()) > 0)
        
        if active_posts == 0:
            return 0.0
            
        return min(1.0, total_engagement / (active_posts * 10))  # Normalize to 0-1 range
    
    def _calculate_theme_stability(self, thread: Dict) -> float:
        """Calculate theme stability (how consistent the theme has been)."""
        history = thread.get("theme_history", [])
        if len(history) <= 1:
            return 1.0  # Perfect stability with only one theme
        
        # Count changes in theme
        changes = sum(1 for i in range(1, len(history)) 
                     if history[i]["theme"] != history[i-1]["theme"])
        
        # Calculate stability as inverse of change rate
        return max(0.0, 1.0 - (changes / len(history)))
    
    def _calculate_age_days(self, thread: Dict) -> float:
        """Calculate age of the thread in days."""
        try:
            created = datetime.fromisoformat(thread["created_at"])
            now = datetime.now()
            return (now - created).total_seconds() / 86400  # Convert seconds to days
        except Exception:
            return 0.0
    
    def _theme_similarity(self, theme1: str, theme2: str) -> float:
        """
        Calculate similarity between two themes.
        
        This is a simple implementation based on word overlap.
        In a real implementation, use semantic similarity with embeddings.
        """
        if not theme1 or not theme2:
            return 0.0
            
        # Convert to lowercase and split into words
        words1 = set(theme1.lower().split())
        words2 = set(theme2.lower().split())
        
        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union

# Singleton instance
_thread_map_instance = None

def get_thread_map(map_path: Optional[str] = None) -> ThreadMap:
    """
    Get or create singleton instance of the thread map.
    
    Args:
        map_path: Optional path to thread map file
        
    Returns:
        ThreadMap instance
    """
    global _thread_map_instance
    
    if _thread_map_instance is None:
        _thread_map_instance = ThreadMap(map_path)
        
    return _thread_map_instance

if __name__ == "__main__":
    # Example usage
    thread_map = get_thread_map()
    
    # Create a new thread
    thread_id = thread_map.create_thread(
        title="Response to Quantum Resonance Theory",
        platform="twitter",
        source_url="https://twitter.com/user/status/123456789",
        description="Thread analyzing community response to quantum theory post",
        initial_theme="quantum consciousness",
        tags=["science", "quantum", "consciousness"],
        phi_impact=0.03
    )
    
    # Add posts to the thread
    thread_map.add_post(
        thread_id=thread_id,
        content="Amazing insights on quantum consciousness! The 3:1 ratio perfectly explains the coherence patterns we see in social systems.",
        author="@quantum_enthusiast",
        engagement={"likes": 15, "shares": 5, "comments": 3},
        sentiment=0.8,
        phi_impact=0.02,
        resonance_delta=0.1
    )
    
    # Update thread theme based on emergent patterns
    thread_map.update_thread_theme(
        thread_id=thread_id,
        new_theme="quantum social dynamics",
        reason="Conversation shifted from pure consciousness to social applications"
    )
    
    print(f"Created example thread with ID: {thread_id}")