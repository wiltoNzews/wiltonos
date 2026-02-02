#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conector de Memória de Coerência

Este módulo conecta o verificador de coerência ao banco de dados vetorial Qdrant,
permitindo o armazenamento e recuperação de estados de coerência para
manter o equilíbrio 3:1 (75% coerência, 25% exploração).

O conector captura automaticamente snapshots de estados de coerência e 
os armazena como vetores, possibilitando busca por similaridade para
informar ajustes no Model Selector baseados em experiência histórica.
"""

import os
import sys
import time
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Importar cliente do Qdrant
from .qdrant_client import get_memory_instance

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/coherence_memory.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("coherence_memory_connector")

class CoherenceMemoryConnector:
    """
    Conecta o verificador de coerência com o sistema de memória vetorial.
    
    Gerencia o ciclo de vida da captura de estados, indexação e recuperação
    para ajudar a manter o equilíbrio 3:1 entre coerência e exploração.
    """
    
    def __init__(self, 
                 coherence_checker_url: str = "http://localhost:5050",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 capture_interval: int = 60,  # segundos
                 auto_start: bool = False):
        """
        Inicializa o conector de memória de coerência.
        
        Args:
            coherence_checker_url: URL do serviço de verificação de coerência
            qdrant_host: Host do Qdrant
            qdrant_port: Porta do Qdrant
            capture_interval: Intervalo entre capturas de estado em segundos
            auto_start: Iniciar captura automática na inicialização
        """
        self.coherence_checker_url = coherence_checker_url
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.capture_interval = capture_interval
        
        # Inicializar cliente Qdrant
        try:
            self.memory = get_memory_instance(host=qdrant_host, port=qdrant_port)
            self.ready = self.memory.ready
            
            if self.ready:
                logger.info(f"Conectado ao sistema de memória Qdrant em {qdrant_host}:{qdrant_port}")
            else:
                logger.error(f"Falha ao conectar ao Qdrant em {qdrant_host}:{qdrant_port}")
        except Exception as e:
            logger.error(f"Erro ao inicializar cliente Qdrant: {str(e)}")
            self.memory = None
            self.ready = False
            
        # Estado da captura automática
        self.capture_task = None
        self.is_capturing = False
        
        # Iniciar captura automática se solicitado
        if auto_start and self.ready:
            self.start_auto_capture()
            
    async def get_coherence_state(self) -> Optional[Dict[str, Any]]:
        """
        Obtém o estado atual de coerência do verificador.
        
        Returns:
            Estado de coerência ou None em caso de erro
        """
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.coherence_checker_url}/coherence",
                    timeout=5
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "success":
                            return data
                    logger.warning(f"Resposta inválida ao obter coerência: {response.status}")
                    return None
        except Exception as e:
            logger.warning(f"Erro ao obter estado de coerência: {str(e)}")
            return None
            
    async def get_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Obtém métricas atuais do sistema.
        
        Returns:
            Métricas ou None em caso de erro
        """
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.coherence_checker_url}/metrics",
                    timeout=5
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "success":
                            return data.get("metrics", {})
                    logger.warning(f"Resposta inválida ao obter métricas: {response.status}")
                    return None
        except Exception as e:
            logger.warning(f"Erro ao obter métricas: {str(e)}")
            return None
            
    async def capture_coherence_snapshot(self) -> Optional[str]:
        """
        Captura um snapshot do estado atual de coerência e armazena no Qdrant.
        
        Returns:
            ID do snapshot ou None em caso de erro
        """
        if not self.ready:
            logger.warning("Sistema de memória não está pronto")
            return None
            
        # Obter estado de coerência
        coherence_state = await self.get_coherence_state()
        if not coherence_state:
            logger.warning("Não foi possível obter estado de coerência")
            return None
            
        # Extrair valores-chave
        phi = coherence_state.get("phi", 0)
        status = coherence_state.get("status", "unknown")
        
        # Obter métricas adicionais
        metrics = await self.get_metrics()
        weights = metrics.get("weights", {}) if metrics else {}
        models = metrics.get("models", {}) if metrics else {}
        recent_prompts = metrics.get("recent_prompts", []) if metrics else []
        
        # Montar metadados
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "weights": weights,
            "models": models,
            "recent_prompts": recent_prompts,
            "raw_state": coherence_state
        }
        
        # Armazenar no Qdrant
        try:
            snapshot_id = self.memory.store_coherence_snapshot(
                phi=phi,
                status=status,
                metadata=metadata
            )
            logger.info(f"Snapshot de coerência armazenado: {snapshot_id} (phi={phi:.4f}, status={status})")
            return snapshot_id
        except Exception as e:
            logger.error(f"Erro ao armazenar snapshot de coerência: {str(e)}")
            return None
            
    async def find_similar_states(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Encontra estados históricos similares ao estado atual.
        
        Args:
            limit: Número máximo de resultados
            
        Returns:
            Lista de estados similares
        """
        if not self.ready:
            logger.warning("Sistema de memória não está pronto")
            return []
            
        # Obter estado atual
        coherence_state = await self.get_coherence_state()
        if not coherence_state:
            logger.warning("Não foi possível obter estado atual de coerência")
            return []
            
        # Extrair valores-chave
        phi = coherence_state.get("phi", 0)
        status = coherence_state.get("status", "unknown")
        
        # Obter métricas adicionais para contexto
        metrics = await self.get_metrics()
        metadata = {}
        if metrics:
            metadata["models"] = metrics.get("models", {})
            metadata["recent_prompts"] = metrics.get("recent_prompts", [])
            
        # Buscar estados similares
        try:
            similar_states = self.memory.find_similar_coherence_states(
                phi=phi,
                status=status,
                metadata=metadata,
                limit=limit
            )
            logger.info(f"Encontrados {len(similar_states)} estados similares ao atual (phi={phi:.4f})")
            return similar_states
        except Exception as e:
            logger.error(f"Erro ao buscar estados similares: {str(e)}")
            return []
            
    async def store_exploration_event(self, 
                                     prompt: str,
                                     event_type: str = "exploration",
                                     model: str = "unknown") -> Optional[str]:
        """
        Armazena um evento de exploração na memória.
        
        Args:
            prompt: Prompt que gerou o evento
            event_type: Tipo de evento de exploração
            model: Modelo utilizado
            
        Returns:
            ID do evento ou None em caso de erro
        """
        if not self.ready:
            logger.warning("Sistema de memória não está pronto")
            return None
            
        # Obter métricas atuais para contexto
        metrics = await self.get_metrics()
        metadata = {}
        if metrics:
            metadata["weights"] = metrics.get("weights", {})
            metadata["coherence"] = metrics.get("coherence", {})
            
        # Obter estado de coerência
        coherence_state = await self.get_coherence_state()
        if coherence_state:
            metadata["phi"] = coherence_state.get("phi", 0)
            metadata["phi_status"] = coherence_state.get("status", "unknown")
            
        # Armazenar evento
        try:
            event_id = self.memory.store_exploration_event(
                event_type=event_type,
                prompt=prompt,
                model=model,
                metadata=metadata
            )
            logger.info(f"Evento de exploração armazenado: {event_id}")
            return event_id
        except Exception as e:
            logger.error(f"Erro ao armazenar evento de exploração: {str(e)}")
            return None
            
    async def auto_capture_loop(self):
        """Loop de captura automática de snapshots de coerência."""
        self.is_capturing = True
        
        try:
            logger.info(f"Iniciando captura automática de estados de coerência a cada {self.capture_interval} segundos")
            
            while self.is_capturing:
                # Capturar snapshot
                await self.capture_coherence_snapshot()
                
                # Aguardar próximo ciclo
                await asyncio.sleep(self.capture_interval)
                
        except asyncio.CancelledError:
            logger.info("Captura automática de estados de coerência interrompida")
            self.is_capturing = False
        except Exception as e:
            logger.error(f"Erro no loop de captura: {str(e)}")
            self.is_capturing = False
            
    def start_auto_capture(self):
        """Inicia a captura automática de snapshots de coerência."""
        if self.capture_task is not None and not self.capture_task.done():
            logger.warning("Captura automática já está em execução")
            return
            
        if not self.ready:
            logger.error("Sistema de memória não está pronto. Não é possível iniciar captura automática")
            return
            
        # Criar e iniciar tarefa de captura
        self.capture_task = asyncio.create_task(self.auto_capture_loop())
        logger.info("Captura automática iniciada")
        
    def stop_auto_capture(self):
        """Interrompe a captura automática de snapshots de coerência."""
        if self.capture_task is None or self.capture_task.done():
            logger.warning("Captura automática não está em execução")
            return
            
        # Cancelar tarefa
        self.capture_task.cancel()
        self.is_capturing = False
        logger.info("Captura automática interrompida")

    async def get_similar_prompt_recommendations(self, prompt: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Busca prompts similares e suas recomendações de modelos.
        
        Args:
            prompt: Prompt atual
            limit: Número máximo de resultados
            
        Returns:
            Lista de eventos com prompts similares
        """
        if not self.ready:
            logger.warning("Sistema de memória não está pronto")
            return []
            
        try:
            similar_events = self.memory.find_similar_exploration_events(
                prompt=prompt,
                limit=limit
            )
            
            # Extrair apenas as informações relevantes
            recommendations = []
            for event in similar_events:
                recommendations.append({
                    "prompt": event.get("prompt", ""),
                    "model": event.get("model", "unknown"),
                    "similarity": event.get("similarity", 0),
                    "metadata": {
                        "phi": event.get("metadata", {}).get("phi", 0),
                        "phi_status": event.get("metadata", {}).get("phi_status", "unknown"),
                        "timestamp": event.get("timestamp", "")
                    }
                })
                
            logger.info(f"Encontrados {len(recommendations)} prompts similares")
            return recommendations
        except Exception as e:
            logger.error(f"Erro ao buscar prompts similares: {str(e)}")
            return []
            
    async def recommend_model_for_prompt(self, prompt: str) -> Optional[str]:
        """
        Recomenda um modelo com base em prompts históricos similares.
        
        Args:
            prompt: Prompt para recomendar modelo
            
        Returns:
            Nome do modelo recomendado ou None
        """
        if not self.ready:
            logger.warning("Sistema de memória não está pronto")
            return None
            
        # Buscar prompts similares
        similar_prompts = await self.get_similar_prompt_recommendations(prompt, limit=5)
        
        if not similar_prompts:
            logger.info("Nenhum prompt similar encontrado para recomendação")
            return None
            
        # Analisar os modelos mais utilizados para prompts similares
        models = {}
        for item in similar_prompts:
            model = item.get("model", "unknown")
            similarity = item.get("similarity", 0)
            
            # Ponderar pela similaridade
            if model in models:
                models[model] += similarity
            else:
                models[model] = similarity
                
        # Encontrar o modelo com maior score ponderado
        if not models:
            return None
            
        recommended_model = max(models.items(), key=lambda x: x[1])[0]
        logger.info(f"Modelo recomendado para o prompt: {recommended_model}")
        
        return recommended_model

# Criar instância singleton
_connector_instance = None

def get_connector_instance(
    coherence_checker_url: str = "http://localhost:5050",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    reinitialize: bool = False
) -> CoherenceMemoryConnector:
    """
    Obtém ou cria instância singleton do conector de memória.
    
    Args:
        coherence_checker_url: URL do verificador de coerência
        qdrant_host: Host do Qdrant
        qdrant_port: Porta do Qdrant
        reinitialize: Forçar reinicialização da instância
        
    Returns:
        Instância do conector
    """
    global _connector_instance
    
    if _connector_instance is None or reinitialize:
        _connector_instance = CoherenceMemoryConnector(
            coherence_checker_url=coherence_checker_url,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port
        )
        
    return _connector_instance