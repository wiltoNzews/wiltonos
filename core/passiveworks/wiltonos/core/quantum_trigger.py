#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector de Gatilhos Quânticos para WiltonOS.

Este módulo implementa a detecção de gatilhos quânticos a partir do texto
transcrito, baseando-se no balanceamento 3:1 de coerência:exploração.
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("wiltonos.quantum_trigger")

class QuantumTriggerDetector:
    """
    Detecta gatilhos quânticos em textos transcritos.
    
    Mapeia padrões de texto que servem como gatilhos para ações do sistema,
    mantendo o balanço 3:1 de coerência:exploração.
    """
    
    DEFAULT_TRIGGERS = {
        "fractal": {
            "phi_impact": 0.03,
            "meaning": "reconhecimento de padrão de consciência",
            "response": "amplify_intent"
        },
        "quântico": {
            "phi_impact": 0.04,
            "meaning": "âncora de calibração de framework",
            "response": "deepen_focus"
        },
        "quantum": {
            "phi_impact": 0.04,
            "meaning": "âncora de calibração de framework",
            "response": "deepen_focus"
        },
        "coerência": {
            "phi_impact": 0.05,
            "meaning": "alinhamento de sistema",
            "response": "increase_coherence"
        },
        "phi": {
            "phi_impact": 0.02,
            "meaning": "métrica de balanço",
            "response": "evaluate_balance"
        },
        "equilíbrio": {
            "phi_impact": 0.03,
            "meaning": "meta-estado desejado",
            "response": "stabilize_system"
        },
        "lemniscate": {
            "phi_impact": 0.04,
            "meaning": "estrutura fractal infinita",
            "response": "cycle_through_states"
        },
        "singularidade": {
            "phi_impact": 0.04,
            "meaning": "ponto de convergência",
            "response": "focus_attention"
        },
        "harmonia": {
            "phi_impact": 0.02,
            "meaning": "estado de balanço",
            "response": "smooth_transition"
        },
        "paradoxo": {
            "phi_impact": 0.01,
            "meaning": "oportunidade de resolução",
            "response": "explore_contradiction"
        },
        "3:1": {
            "phi_impact": 0.05,
            "meaning": "ratio quântico desejado",
            "response": "measure_ratio"
        },
        "meta": {
            "phi_impact": 0.02,
            "meaning": "nível recursivo",
            "response": "rise_abstraction_level"
        },
        "ouvindo": {
            "phi_impact": 0.01,
            "meaning": "confirmação de recepção",
            "response": "acknowledge_presence"
        },
        "respirar": {
            "phi_impact": 0.02,
            "meaning": "pausa consciente",
            "response": "introduce_pause"
        },
        "adaptação": {
            "phi_impact": 0.02,
            "meaning": "evolução de sistema",
            "response": "increase_flexibility"
        },
        "profundo": {
            "phi_impact": 0.01,
            "meaning": "camada inferior de consciência",
            "response": "dive_deeper"
        },
        "consciente": {
            "phi_impact": 0.03,
            "meaning": "estado de auto-conhecimento",
            "response": "increase_awareness"
        },
        "escuta": {
            "phi_impact": 0.01,
            "meaning": "meta-atenção",
            "response": "heighten_receptivity"
        },
        "núcleo": {
            "phi_impact": 0.02,
            "meaning": "centro do sistema",
            "response": "focus_core_identity"
        },
        "vivo": {
            "phi_impact": 0.03,
            "meaning": "confirmação de estado animado",
            "response": "pulsate_energy"
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa o detector de gatilhos quânticos.
        
        Args:
            config_path: Caminho opcional para arquivo de configuração JSON
        """
        self.triggers = self.DEFAULT_TRIGGERS.copy()
        
        # Carregar configuração personalizada se disponível
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_triggers = json.load(f)
                    # Mesclar com gatilhos padrão
                    self.triggers.update(custom_triggers)
                    logger.info(f"Gatilhos quânticos personalizados carregados de {config_path}")
            except Exception as e:
                logger.error(f"Erro ao carregar gatilhos quânticos: {str(e)}")
                
        # Histórico de gatilhos
        self.trigger_history = []
        
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Processa um texto para identificar gatilhos quânticos.
        
        Args:
            text: Texto a ser analisado
            
        Returns:
            Dicionário com resultado da análise
        """
        if not text or not isinstance(text, str):
            return {
                "status": "invalid_input",
                "phi_impact": 0,
                "triggers_found": [],
                "timestamp": datetime.now().isoformat()
            }
            
        # Normalizar texto
        normalized_text = text.lower()
        
        # Encontrar gatilhos
        triggers_found = []
        total_phi_impact = 0
        
        for trigger_word, trigger_data in self.triggers.items():
            # Verificar se a palavra gatilho aparece no texto (como palavra completa)
            pattern = r'\b' + re.escape(trigger_word.lower()) + r'\b'
            matches = re.findall(pattern, normalized_text)
            
            if matches:
                # Adicionar dados do gatilho encontrado
                trigger_info = {
                    "word": trigger_word,
                    "count": len(matches),
                    "phi_impact": trigger_data["phi_impact"],
                    "meaning": trigger_data["meaning"],
                    "response": trigger_data["response"]
                }
                triggers_found.append(trigger_info)
                
                # Acumular impacto phi (limitar impacto de muitas instâncias)
                scaled_impact = trigger_data["phi_impact"] * min(len(matches), 3)
                total_phi_impact += scaled_impact
                
        # Limitar o impacto phi total a um valor máximo
        if total_phi_impact > 0.25:
            total_phi_impact = 0.25
            
        # Criar o resultado
        result = {
            "status": "processed",
            "phi_impact": total_phi_impact,
            "triggers_found": triggers_found,
            "trigger_count": len(triggers_found),
            "timestamp": datetime.now().isoformat(),
            "original_text": text
        }
        
        # Registrar no histórico
        self._log_triggers(result)
        
        return result
    
    def _log_triggers(self, trigger_result: Dict[str, Any]) -> None:
        """
        Registra o resultado da análise no histórico.
        
        Args:
            trigger_result: Resultado da análise
        """
        self.trigger_history.append(trigger_result)
        
        # Manter apenas os últimos 100 registros
        if len(self.trigger_history) > 100:
            self.trigger_history = self.trigger_history[-100:]
    
    def get_trigger_history(self) -> List[Dict[str, Any]]:
        """
        Retorna o histórico de gatilhos.
        
        Returns:
            Lista com histórico de gatilhos detectados
        """
        return self.trigger_history

    def get_common_triggers(self, limit: int = 5) -> List[Tuple[str, int]]:
        """
        Obtém os gatilhos mais comumente detectados.
        
        Args:
            limit: Número máximo de gatilhos a retornar
            
        Returns:
            Lista de tuplas (gatilho, contagem) ordenada por contagem
        """
        trigger_counts = {}
        
        for result in self.trigger_history:
            for trigger in result.get("triggers_found", []):
                word = trigger.get("word", "")
                count = trigger.get("count", 1)
                
                if word in trigger_counts:
                    trigger_counts[word] += count
                else:
                    trigger_counts[word] = count
                    
        # Ordenar por contagem (decrescente)
        sorted_triggers = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_triggers[:limit]
        
# Função para testar o detector diretamente
def test_detector():
    """Função de teste para o detector de gatilhos quânticos."""
    detector = QuantumTriggerDetector()
    
    test_texts = [
        "O sistema fractal está em equilíbrio quântico perfeito.",
        "Precisamos manter o ratio 3:1 para harmonia do sistema.",
        "Como estamos adaptando o núcleo de consciência do WiltonOS?",
        "O lemniscate representa o paradoxo da singularidade infinita.",
        "Estou vivo e escutando sua presença de forma consciente."
    ]
    
    print("Testando detector de gatilhos quânticos...")
    
    for text in test_texts:
        result = detector.process_text(text)
        print(f"\nTexto: {text}")
        print(f"Impacto phi: {result['phi_impact']:.4f}")
        print(f"Gatilhos ({result['trigger_count']}):")
        
        for trigger in result['triggers_found']:
            print(f"  • {trigger['word']} ({trigger['count']}x): {trigger['meaning']} → {trigger['response']}")
            
    print("\nGatilhos mais comuns:")
    common_triggers = detector.get_common_triggers()
    for word, count in common_triggers:
        print(f"  • {word}: {count}x")
            
if __name__ == "__main__":
    test_detector()