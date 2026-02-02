"""
Módulo de memória vetorial para WiltonOS.

Este módulo implementa:
1. Integração com Qdrant para memória vetorial de longo prazo
2. Armazenamento de estados de coerência e exploração
3. Recuperação baseada em similaridade para informar decisões do Model Selector
4. Ciclo de feedback inteligente para manter o balanceamento 3:1
5. Monitoramento de saúde e feedback para o operador do sistema
6. Sistema de registro alimentar com análise de impacto em phi
"""

from .qdrant_client import get_memory_instance, QdrantMemory
from .coherence_memory_connector import get_connector_instance, CoherenceMemoryConnector
from .health_hooks_connector import get_health_hooks_instance, HealthHooksConnector
from .food_logging import get_food_logger_instance, FoodLogger

__all__ = [
    "get_memory_instance", 
    "QdrantMemory",
    "get_connector_instance",
    "CoherenceMemoryConnector",
    "get_health_hooks_instance",
    "HealthHooksConnector",
    "get_food_logger_instance",
    "FoodLogger"
]