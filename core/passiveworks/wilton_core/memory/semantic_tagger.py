"""
Semantic Tagger for WiltonOS
----------------------------
Provides automatic semantic tagging for entries based on content analysis.
Extracts themes, emotions, and concepts with associated phi impact values.
"""

import re
import json
import os
from typing import List, Dict, Any, Optional, Tuple

# Default semantic tag mappings
DEFAULT_TAG_MAPPINGS_PATH = os.path.join(os.path.dirname(__file__), 'semantic_tags.json')

class SemanticTagger:
    """
    Analyzes text content and automatically applies semantic tags with phi impact scores.
    """
    
    def __init__(self, mappings_path: Optional[str] = None):
        """
        Initialize semantic tagger with tag mappings.
        
        Args:
            mappings_path: Path to JSON file with tag mappings
        """
        self.mappings_path = mappings_path or DEFAULT_TAG_MAPPINGS_PATH
        self.tag_mappings = self._load_mappings()
    
    def _load_mappings(self) -> Dict:
        """Load semantic tag mappings from file."""
        # Default mappings if file doesn't exist
        default_mappings = {
            "emotional": {
                "vida": {"tag": "life", "phi_impact": 0.07, "valence": "positive"},
                "perda": {"tag": "loss", "phi_impact": -0.04, "valence": "negative"},
                "identidade": {"tag": "identity", "phi_impact": 0.05, "valence": "neutral"},
                "mar": {"tag": "ocean", "phi_impact": 0.06, "valence": "positive"},
                "água": {"tag": "water", "phi_impact": 0.05, "valence": "positive"},
                "ódio": {"tag": "hate", "phi_impact": -0.08, "valence": "negative"},
                "amor": {"tag": "love", "phi_impact": 0.08, "valence": "positive"},
                "vitória": {"tag": "victory", "phi_impact": 0.06, "valence": "positive"},
                "derrota": {"tag": "defeat", "phi_impact": -0.04, "valence": "negative"},
                "medo": {"tag": "fear", "phi_impact": -0.05, "valence": "negative"},
                "esperança": {"tag": "hope", "phi_impact": 0.07, "valence": "positive"},
                "felicidade": {"tag": "happiness", "phi_impact": 0.08, "valence": "positive"},
                "tristeza": {"tag": "sadness", "phi_impact": -0.06, "valence": "negative"},
                "paz": {"tag": "peace", "phi_impact": 0.05, "valence": "positive"},
                "raiva": {"tag": "anger", "phi_impact": -0.07, "valence": "negative"}
            },
            "conceptual": {
                "quântico": {"tag": "quantum", "phi_impact": 0.04, "domain": "science"},
                "fractal": {"tag": "fractal", "phi_impact": 0.03, "domain": "mathematics"},
                "consciência": {"tag": "consciousness", "phi_impact": 0.05, "domain": "philosophy"},
                "ressonância": {"tag": "resonance", "phi_impact": 0.04, "domain": "physics"},
                "equilíbrio": {"tag": "balance", "phi_impact": 0.03, "domain": "philosophy"},
                "coerência": {"tag": "coherence", "phi_impact": 0.04, "domain": "systems"},
                "sistema": {"tag": "system", "phi_impact": 0.02, "domain": "technology"},
                "mente": {"tag": "mind", "phi_impact": 0.04, "domain": "psychology"},
                "padrão": {"tag": "pattern", "phi_impact": 0.03, "domain": "science"},
                "caos": {"tag": "chaos", "phi_impact": -0.03, "domain": "mathematics"},
                "ordem": {"tag": "order", "phi_impact": 0.02, "domain": "philosophy"},
                "iemanjá": {"tag": "spiritual_deity", "phi_impact": 0.08, "domain": "spirituality"},
                "religião": {"tag": "religion", "phi_impact": 0.01, "domain": "spirituality"},
                "ciência": {"tag": "science", "phi_impact": 0.02, "domain": "knowledge"}
            },
            "social": {
                "publicar": {"tag": "publish", "phi_impact": 0.01, "context": "social_media"},
                "compartilhar": {"tag": "share", "phi_impact": 0.02, "context": "social_media"},
                "tweet": {"tag": "tweet", "phi_impact": 0.01, "context": "twitter"},
                "postar": {"tag": "post", "phi_impact": 0.01, "context": "social_media"},
                "viral": {"tag": "viral", "phi_impact": 0.03, "context": "social_media"},
                "seguidor": {"tag": "follower", "phi_impact": 0.01, "context": "social_media"},
                "comentário": {"tag": "comment", "phi_impact": 0.02, "context": "social_media"},
                "resposta": {"tag": "response", "phi_impact": 0.02, "context": "interaction"},
                "crítica": {"tag": "criticism", "phi_impact": -0.02, "context": "feedback"},
                "elogio": {"tag": "praise", "phi_impact": 0.03, "context": "feedback"},
                "comunidade": {"tag": "community", "phi_impact": 0.03, "context": "belonging"},
                "grupo": {"tag": "group", "phi_impact": 0.01, "context": "belonging"},
                "debate": {"tag": "debate", "phi_impact": 0.02, "context": "discussion"},
                "conflito": {"tag": "conflict", "phi_impact": -0.04, "context": "interaction"},
                "ironia": {"tag": "irony", "phi_impact": 0.01, "context": "expression"},
                "silêncio": {"tag": "silence", "phi_impact": -0.01, "context": "expression"},
                "provocação": {"tag": "provocation", "phi_impact": -0.02, "context": "interaction"}
            }
        }
        
        # Try to load from file
        if os.path.exists(self.mappings_path):
            try:
                with open(self.mappings_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading semantic tag mappings: {e}")
                
        # Save default mappings if file doesn't exist
        try:
            os.makedirs(os.path.dirname(self.mappings_path), exist_ok=True)
            with open(self.mappings_path, 'w', encoding='utf-8') as f:
                json.dump(default_mappings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving default semantic tag mappings: {e}")
        
        return default_mappings
    
    def tag_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze content and extract semantic tags with phi impact.
        
        Args:
            text: Content to analyze
            
        Returns:
            Dictionary with extracted tags and metadata
        """
        if not text or not isinstance(text, str):
            return {"tags": [], "phi_impact": 0, "emotional_valence": "neutral"}
        
        # Normalize text
        text = text.lower()
        
        # Initialize results
        extracted_tags = []
        total_phi_impact = 0
        
        # Check for words in each category
        for category, mappings in self.tag_mappings.items():
            for key_word, data in mappings.items():
                # Find whole word matches
                pattern = r'\b' + re.escape(key_word) + r'\b'
                matches = re.findall(pattern, text)
                
                if matches:
                    # Add tag data
                    tag_data = {
                        "tag": data["tag"],
                        "category": category,
                        "count": len(matches),
                        "phi_impact": data["phi_impact"]
                    }
                    
                    # Add category-specific metadata
                    if category == "emotional" and "valence" in data:
                        tag_data["valence"] = data["valence"]
                    elif category == "conceptual" and "domain" in data:
                        tag_data["domain"] = data["domain"]
                    elif category == "social" and "context" in data:
                        tag_data["context"] = data["context"]
                    
                    extracted_tags.append(tag_data)
                    
                    # Accumulate phi impact (multiple occurrences have diminishing returns)
                    phi_impact = data["phi_impact"] * min(len(matches), 3) * 0.6
                    total_phi_impact += phi_impact
        
        # Determine overall emotional valence
        emotional_valence = self._determine_emotional_valence(extracted_tags)
        
        # Limit total phi impact to reasonable range
        total_phi_impact = max(-0.3, min(0.3, total_phi_impact))
        
        # Build result
        result = {
            "tags": [tag["tag"] for tag in extracted_tags],
            "detailed_tags": extracted_tags,
            "phi_impact": round(total_phi_impact, 2),
            "emotional_valence": emotional_valence
        }
        
        return result
    
    def _determine_emotional_valence(self, tags: List[Dict]) -> str:
        """Determine overall emotional valence from tags."""
        # Filter for emotional tags
        emotional_tags = [tag for tag in tags if tag.get("category") == "emotional"]
        
        if not emotional_tags:
            return "neutral"
        
        # Count valence types
        valence_counts = {"positive": 0, "negative": 0, "neutral": 0}
        
        for tag in emotional_tags:
            valence = tag.get("valence", "neutral")
            valence_counts[valence] += tag.get("count", 1)
        
        # Determine dominant valence
        if valence_counts["positive"] > valence_counts["negative"] + valence_counts["neutral"]:
            return "positive"
        elif valence_counts["negative"] > valence_counts["positive"] + valence_counts["neutral"]:
            return "negative"
        elif valence_counts["positive"] > 0 and valence_counts["negative"] > 0:
            return "complex"
        else:
            return "neutral"
    
    def add_tag_mapping(self, word: str, category: str, tag: str, phi_impact: float, 
                      metadata: Optional[Dict] = None) -> bool:
        """
        Add a new tag mapping.
        
        Args:
            word: Keyword to match
            category: Category (emotional, conceptual, social)
            tag: Tag to apply
            phi_impact: Impact on phi level
            metadata: Additional metadata for the tag
            
        Returns:
            Success status
        """
        if category not in self.tag_mappings:
            self.tag_mappings[category] = {}
        
        # Create mapping data
        mapping = {
            "tag": tag,
            "phi_impact": phi_impact
        }
        
        # Add category-specific metadata
        if metadata:
            for key, value in metadata.items():
                mapping[key] = value
        
        # Add to mappings
        self.tag_mappings[category][word] = mapping
        
        # Save updated mappings
        try:
            with open(self.mappings_path, 'w', encoding='utf-8') as f:
                json.dump(self.tag_mappings, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving semantic tag mappings: {e}")
            return False
    
    def remove_tag_mapping(self, word: str, category: str) -> bool:
        """
        Remove a tag mapping.
        
        Args:
            word: Word to remove
            category: Category of the word
            
        Returns:
            Success status
        """
        if category in self.tag_mappings and word in self.tag_mappings[category]:
            del self.tag_mappings[category][word]
            
            # Save updated mappings
            try:
                with open(self.mappings_path, 'w', encoding='utf-8') as f:
                    json.dump(self.tag_mappings, f, indent=2, ensure_ascii=False)
                return True
            except Exception as e:
                print(f"Error saving semantic tag mappings: {e}")
        
        return False

# Singleton pattern
_tagger_instance = None

def get_semantic_tagger(mappings_path: Optional[str] = None) -> SemanticTagger:
    """
    Get or create singleton instance of the semantic tagger.
    
    Args:
        mappings_path: Optional path to tag mappings file
        
    Returns:
        SemanticTagger instance
    """
    global _tagger_instance
    
    if _tagger_instance is None:
        _tagger_instance = SemanticTagger(mappings_path)
        
    return _tagger_instance

if __name__ == "__main__":
    # Example usage
    tagger = get_semantic_tagger()
    
    # Example text
    text = "A água do mar representa o equilíbrio fractal da consciência quântica. Iemanjá me guia nessa jornada de ressonância, superando o medo e encontrando esperança. Vou compartilhar esse insight nas redes sociais."
    
    # Tag the content
    result = tagger.tag_content(text)
    
    print("Extracted tags:", ", ".join(result["tags"]))
    print("Phi impact:", result["phi_impact"])
    print("Emotional valence:", result["emotional_valence"])
    print("\nDetailed tags:")
    for tag in result["detailed_tags"]:
        print(f"- {tag['tag']} (category: {tag['category']}, count: {tag['count']}, phi: {tag['phi_impact']})")