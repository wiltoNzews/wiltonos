#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conector Health-Hooks para Memória Vetorial

Este módulo integra o sistema Health-Hooks do WiltonOS com o banco de dados
vetorial Qdrant, permitindo o armazenamento e análise de eventos de saúde
para recomendações adaptativas e alertas personalizados.

Características principais:
1. Registro automático de eventos de saúde como vetores
2. Busca por similaridade para identificar padrões recorrentes
3. Alertas baseados em frequência e correlação de eventos
4. Recomendações personalizadas com base em histórico

Este sistema ajuda a manter o equilíbrio não apenas do sistema, mas também
do operador, contribuindo para a coerência global do WiltonOS.
"""

import os
import sys
import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

# Importar cliente Qdrant
from .qdrant_client import get_memory_instance

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/health_hooks_memory.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("health_hooks_connector")

# Configurações padrão
HEALTH_HOOKS_COLLECTION = "health_hooks_memory"
HEALTH_ALERT_THRESHOLD = 2  # Número de eventos dentro do período para gerar alerta
HEALTH_ALERT_PERIOD = 24  # Período em horas para considerar eventos

class HealthHooksConnector:
    """
    Conecta o sistema Health-Hooks com o banco de dados vetorial.
    
    Permite armazenar eventos de saúde, buscar padrões, e gerar alertas
    baseados em frequência e similaridade de eventos.
    """
    
    def __init__(self,
                 health_hooks_url: str = "http://localhost:5050/health-hooks",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 alert_threshold: int = HEALTH_ALERT_THRESHOLD,
                 alert_period: int = HEALTH_ALERT_PERIOD,
                 auto_start: bool = False):
        """
        Inicializa o conector Health-Hooks.
        
        Args:
            health_hooks_url: URL do serviço Health-Hooks
            qdrant_host: Host do Qdrant
            qdrant_port: Porta do Qdrant
            alert_threshold: Número de eventos para gerar alerta
            alert_period: Período (horas) para considerar eventos
            auto_start: Iniciar captura automática na inicialização
        """
        self.health_hooks_url = health_hooks_url
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.alert_threshold = alert_threshold
        self.alert_period = alert_period
        
        # Inicializar cliente Qdrant
        try:
            self.memory = get_memory_instance(host=qdrant_host, port=qdrant_port)
            self.ready = self.memory.ready
            
            if self.ready:
                logger.info(f"Conectado ao sistema de memória Qdrant em {qdrant_host}:{qdrant_port}")
                self._initialize_collection()
            else:
                logger.error(f"Falha ao conectar ao Qdrant em {qdrant_host}:{qdrant_port}")
        except Exception as e:
            logger.error(f"Erro ao inicializar cliente Qdrant: {str(e)}")
            self.memory = None
            self.ready = False
            
        # Estado da captura automática
        self.listener_task = None
        self.is_listening = False
        
        # Iniciar escuta automática se solicitado
        if auto_start and self.ready:
            self.start_auto_listening()
            
    def _initialize_collection(self):
        """Inicializa a coleção para eventos de saúde."""
        try:
            # Verificar se a coleção existe
            collections = self.memory.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if HEALTH_HOOKS_COLLECTION not in collection_names:
                # Criar a coleção
                logger.info(f"Criando coleção {HEALTH_HOOKS_COLLECTION}...")
                self.memory.client.create_collection(
                    collection_name=HEALTH_HOOKS_COLLECTION,
                    vectors_config=self.memory.client.http.models.VectorParams(
                        size=self.memory.vector_size,
                        distance=self.memory.client.http.models.Distance.COSINE
                    )
                )
                # Metadata não é suportado como parâmetro direto na criação da coleção
                # na versão atual do cliente Qdrant
                logger.info(f"Coleção {HEALTH_HOOKS_COLLECTION} criada com sucesso")
            else:
                logger.info(f"Coleção {HEALTH_HOOKS_COLLECTION} já existe")
        except Exception as e:
            logger.error(f"Erro ao inicializar coleção de saúde: {str(e)}")
            
    async def get_health_events(self, 
                              limit: int = 10, 
                              event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtém eventos de saúde do serviço Health-Hooks.
        
        Args:
            limit: Número máximo de eventos a retornar
            event_type: Filtrar por tipo de evento
            
        Returns:
            Lista de eventos de saúde
        """
        import aiohttp
        
        try:
            url = f"{self.health_hooks_url}/events"
            if event_type:
                url += f"?type={event_type}&limit={limit}"
            else:
                url += f"?limit={limit}"
                
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "success":
                            return data.get("events", [])
                    logger.warning(f"Resposta inválida ao obter eventos de saúde: {response.status}")
                    return []
        except Exception as e:
            logger.warning(f"Erro ao obter eventos de saúde: {str(e)}")
            return []
            
    async def store_health_event(self, 
                               event_type: str,
                               event_name: str,
                               notes: str = "",
                               metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Armazena um evento de saúde no Qdrant e no serviço Health-Hooks.
        
        Args:
            event_type: Tipo de evento (ex: "symptom", "measurement", "medication")
            event_name: Nome do evento
            notes: Notas adicionais
            metadata: Metadados adicionais
            
        Returns:
            ID do evento ou None em caso de erro
        """
        if not self.ready:
            logger.warning("Sistema de memória não está pronto")
            return None
            
        # Criar descrição para embedding
        timestamp = datetime.now().isoformat()
        description = f"Type: {event_type}, Name: {event_name}, Time: {timestamp}"
        
        # Adicionar notas se disponíveis
        if notes:
            description += f", Notes: {notes}"
            
        # Texto para embedding
        embedding_text = description
        
        # Gerar embedding
        try:
            embedding = self.memory.get_embedding(embedding_text)
        except ValueError:
            # Fallback para embedding sintético
            logger.warning("Usando embedding sintético para evento de saúde")
            import numpy as np
            rng = np.random.default_rng(seed=hash(embedding_text) % 2**32)
            embedding = rng.random(self.memory.vector_size).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embedding = embedding.tolist()
            
        # Preparar metadados
        if metadata is None:
            metadata = {}
            
        # Adicionar timestamp
        metadata["timestamp"] = timestamp
        
        # Criar ID único
        import uuid
        point_id = str(uuid.uuid4())
        
        # Preparar payload
        point = self.memory.client.http.models.PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "event_type": event_type,
                "event_name": event_name,
                "notes": notes,
                "timestamp": timestamp,
                "metadata": metadata
            }
        )
        
        # Armazenar no Qdrant
        try:
            self.memory.client.upsert(
                collection_name=HEALTH_HOOKS_COLLECTION,
                points=[point]
            )
            logger.info(f"Evento de saúde armazenado: {point_id} - Tipo: {event_type}, Nome: {event_name}")
            
            # Enviar para o serviço Health-Hooks se disponível
            await self._send_to_health_hooks(event_type, event_name, notes, metadata)
            
            return point_id
        except Exception as e:
            logger.error(f"Erro ao armazenar evento de saúde: {str(e)}")
            return None
            
    async def _send_to_health_hooks(self,
                                  event_type: str,
                                  event_name: str,
                                  notes: str,
                                  metadata: Dict[str, Any]) -> bool:
        """
        Envia um evento para o serviço Health-Hooks.
        
        Args:
            event_type: Tipo de evento
            event_name: Nome do evento
            notes: Notas adicionais
            metadata: Metadados adicionais
            
        Returns:
            True se enviado com sucesso, False caso contrário
        """
        import aiohttp
        
        try:
            payload = {
                "type": event_type,
                "name": event_name,
                "notes": notes,
                "metadata": metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.health_hooks_url}/event",
                    json=payload,
                    timeout=5
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("status") == "success"
                    logger.warning(f"Falha ao enviar evento para Health-Hooks: {response.status}")
                    return False
        except Exception as e:
            logger.warning(f"Erro ao enviar para Health-Hooks: {str(e)}")
            return False
            
    async def find_similar_health_events(self,
                                       query: str,
                                       event_type: Optional[str] = None,
                                       limit: int = 5) -> List[Dict[str, Any]]:
        """
        Encontra eventos de saúde similares com base em uma consulta.
        
        Args:
            query: Texto da consulta
            event_type: Filtrar por tipo de evento
            limit: Número máximo de resultados
            
        Returns:
            Lista de eventos similares
        """
        if not self.ready:
            logger.warning("Sistema de memória não está pronto")
            return []
            
        # Gerar embedding
        try:
            query_embedding = self.memory.get_embedding(query)
        except ValueError:
            # Fallback
            logger.warning("Usando embedding sintético para busca de eventos de saúde")
            import numpy as np
            rng = np.random.default_rng(seed=hash(query) % 2**32)
            query_embedding = rng.random(self.memory.vector_size).astype(np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = query_embedding.tolist()
            
        # Filtro para tipo de evento (se especificado)
        filter_param = None
        if event_type:
            filter_param = self.memory.client.http.models.Filter(
                must=[
                    self.memory.client.http.models.FieldCondition(
                        key="event_type",
                        match=self.memory.client.http.models.MatchValue(value=event_type)
                    )
                ]
            )
            
        # Buscar eventos similares
        try:
            search_result = self.memory.client.search(
                collection_name=HEALTH_HOOKS_COLLECTION,
                query_vector=query_embedding,
                limit=limit,
                query_filter=filter_param
            )
            
            # Formatar resultados
            results = []
            for point in search_result:
                results.append({
                    "id": point.id,
                    "event_type": point.payload.get("event_type"),
                    "event_name": point.payload.get("event_name"),
                    "notes": point.payload.get("notes"),
                    "timestamp": point.payload.get("timestamp"),
                    "metadata": point.payload.get("metadata"),
                    "similarity": point.score
                })
                
            logger.info(f"Encontrados {len(results)} eventos de saúde similares à consulta: {query}")
            return results
        except Exception as e:
            logger.error(f"Erro ao buscar eventos similares: {str(e)}")
            return []
            
    async def count_recent_events(self,
                                event_type: Optional[str] = None,
                                event_name: Optional[str] = None,
                                hours: int = 24) -> int:
        """
        Conta eventos recentes de um determinado tipo e/ou nome.
        
        Args:
            event_type: Tipo de evento para filtrar
            event_name: Nome do evento para filtrar
            hours: Período em horas para considerar eventos
            
        Returns:
            Número de eventos encontrados
        """
        if not self.ready:
            logger.warning("Sistema de memória não está pronto")
            return 0
            
        # Calcular timestamp de corte
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        # Construir filtro
        filter_conditions = []
        
        # Filtro de timestamp
        filter_conditions.append(
            self.memory.client.http.models.FieldCondition(
                key="timestamp",
                match=self.memory.client.http.models.MatchValue(value=cutoff_time),
                range=self.memory.client.http.models.Range(gt=None)
            )
        )
        
        # Filtro de tipo de evento (se especificado)
        if event_type:
            filter_conditions.append(
                self.memory.client.http.models.FieldCondition(
                    key="event_type",
                    match=self.memory.client.http.models.MatchValue(value=event_type)
                )
            )
            
        # Filtro de nome de evento (se especificado)
        if event_name:
            filter_conditions.append(
                self.memory.client.http.models.FieldCondition(
                    key="event_name",
                    match=self.memory.client.http.models.MatchValue(value=event_name)
                )
            )
            
        # Criar filtro composto
        filter_param = self.memory.client.http.models.Filter(
            must=filter_conditions
        )
        
        try:
            # Contar documentos
            count_result = self.memory.client.count(
                collection_name=HEALTH_HOOKS_COLLECTION,
                count_filter=filter_param
            )
            
            logger.info(f"Contagem de eventos recentes: {count_result.count} (tipo: {event_type}, nome: {event_name}, período: {hours}h)")
            return count_result.count
        except Exception as e:
            logger.error(f"Erro ao contar eventos recentes: {str(e)}")
            return 0
            
    async def get_coherence_state(self) -> Dict[str, Any]:
        """
        Obtém o estado atual de coerência do sistema.
        
        Returns:
            Um dicionário contendo o estado atual de coerência (phi)
            e outras métricas associadas.
        """
        try:
            # Valores padrão para o estado de coerência
            phi = 0.5  # Valor padrão do phi (equilíbrio 50%)
            dphi_dt = 0.0  # Taxa de mudança padrão
            
            # Se o sistema não está pronto, retorna valores padrão
            # para que outras funções possam continuar funcionando
            if not self.ready:
                logger.warning("Sistema de memória não está pronto, usando valores padrão de coerência")
                status = "transição"
                deviation = phi - 0.75
                return {
                    "phi": phi,
                    "dphi_dt": dphi_dt,
                    "status": status,
                    "deviation": deviation,
                    "target_phi": 0.75,
                    "fallback": True  # Indica que estamos usando um fallback
                }
            
            # Verificar se há eventos recentes que afetam o phi
            recent_events = await self.find_similar_health_events(
                query="coherence state",
                event_type="measurement",
                limit=1
            )
            
            if recent_events and len(recent_events) > 0:
                # Obter valores do evento mais recente
                latest_event = recent_events[0]
                event_meta = latest_event.get("metadata", {})
                
                if "phi" in event_meta:
                    phi = float(event_meta["phi"])
                if "dphi_dt" in event_meta:
                    dphi_dt = float(event_meta["dphi_dt"])
                    
            # Determinar status com base no valor phi
            if phi < 0.25:
                status = "exploração_excessiva"
            elif phi > 0.85:
                status = "supercoerência"
            elif 0.65 <= phi <= 0.85:
                status = "sweet_spot"
            else:
                status = "transição"
                
            # Calcular desvio do valor ideal (0.75)
            deviation = phi - 0.75
                
            # Retornar estado completo
            return {
                "phi": phi,
                "dphi_dt": dphi_dt, 
                "status": status,
                "deviation": deviation,
                "target": 0.75,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Erro ao obter estado de coerência: {str(e)}")
            # Retornar valores padrão em caso de erro
            return {
                "phi": 0.5,
                "dphi_dt": 0.0,
                "status": "unknown",
                "deviation": -0.25,
                "target": 0.75,
                "timestamp": datetime.now().isoformat()
            }
            
    async def check_for_alerts(self, 
                             event_types: Optional[List[str]] = None,
                             event_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Verifica se há alertas baseados em frequência de eventos.
        
        Args:
            event_types: Lista de tipos de eventos para verificar
            event_names: Lista de nomes de eventos para verificar
            
        Returns:
            Lista de alertas gerados
        """
        if not self.ready:
            logger.warning("Sistema de memória não está pronto")
            return []
            
        alerts = []
        
        # Se não foram especificados tipos, usar lista padrão
        if not event_types:
            event_types = ["symptom", "measurement", "medication"]
            
        # Verificar cada tipo
        for event_type in event_types:
            # Se foram especificados nomes, verificar cada um
            if event_names:
                for event_name in event_names:
                    count = await self.count_recent_events(
                        event_type=event_type,
                        event_name=event_name,
                        hours=self.alert_period
                    )
                    
                    # Gerar alerta se acima do threshold
                    if count >= self.alert_threshold:
                        alert = {
                            "event_type": event_type,
                            "event_name": event_name,
                            "count": count,
                            "period_hours": self.alert_period,
                            "threshold": self.alert_threshold,
                            "timestamp": datetime.now().isoformat(),
                            "message": f"Alerta: {count} eventos do tipo '{event_type}' e nome '{event_name}' nas últimas {self.alert_period} horas"
                        }
                        alerts.append(alert)
                        logger.warning(f"ALERTA: {alert['message']}")
            else:
                # Verificar apenas por tipo
                count = await self.count_recent_events(
                    event_type=event_type,
                    hours=self.alert_period
                )
                
                # Gerar alerta se acima do threshold
                if count >= self.alert_threshold:
                    alert = {
                        "event_type": event_type,
                        "count": count,
                        "period_hours": self.alert_period,
                        "threshold": self.alert_threshold,
                        "timestamp": datetime.now().isoformat(),
                        "message": f"Alerta: {count} eventos do tipo '{event_type}' nas últimas {self.alert_period} horas"
                    }
                    alerts.append(alert)
                    logger.warning(f"ALERTA: {alert['message']}")
                    
        return alerts
        
    async def get_recommendations(self, 
                                alert: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Gera recomendações personalizadas com base em um alerta.
        
        Args:
            alert: Alerta gerado pelo sistema
            
        Returns:
            Lista de recomendações
        """
        # Recomendações pré-definidas para tipos específicos
        recommendations = []
        
        event_type = alert.get("event_type")
        event_name = alert.get("event_name")
        
        # Recomendações para sintomas
        if event_type == "symptom":
            if event_name == "coceira no braço":
                recommendations.append({
                    "type": "medication",
                    "name": "anti-histamínico",
                    "message": "Considere tomar um anti-histamínico conforme prescrito."
                })
                recommendations.append({
                    "type": "technique",
                    "name": "respiração diafragmática",
                    "message": "Pratique respiração diafragmática: inspire contando até 4, segure 2s, expire em 6. Repita 5 ciclos."
                })
                
                # Verificar se há alimentos correlacionados
                try:
                    # Buscar eventos de alimentação recentes (últimas 6 horas)
                    recent_food = await self.count_recent_events(
                        event_type="food",
                        hours=6
                    )
                    
                    if recent_food > 0:
                        recommendations.append({
                            "type": "nutrition",
                            "name": "avaliação alimentar",
                            "message": "Foram detectados eventos alimentares nas últimas 6 horas. Considere revisar os alimentos consumidos recentemente."
                        })
                except Exception as e:
                    logger.error(f"Erro ao verificar alimentos correlacionados: {str(e)}")
            else:
                recommendations.append({
                    "type": "action",
                    "name": "consulta médica",
                    "message": "Considere agendar uma consulta médica se os sintomas persistirem."
                })
                
        # Recomendações para medições
        elif event_type == "measurement":
            recommendations.append({
                "type": "technique",
                "name": "pausa consciente",
                "message": "Faça uma pausa consciente: sente firme, pés no chão, descreva 5 objetos ao redor."
            })
            recommendations.append({
                "type": "action",
                "name": "registro",
                "message": "Continue registrando as medições para identificar padrões."
            })
            
        # Recomendações para alimentação
        elif event_type == "food":
            # Verificar phi atual
            coherence_state = await self.get_coherence_state()
            if coherence_state:
                phi = coherence_state.get("phi", 0)
                status = coherence_state.get("status", "unknown")
                
                # Buscar alimentos similares para comparar impacto no phi
                similar_foods = await self.find_similar_health_events(
                    query=event_name,
                    event_type="food",
                    limit=3
                )
                
                # Analisar histórico de phi após consumo deste alimento
                phi_impacts = []
                for food in similar_foods:
                    food_timestamp = datetime.fromisoformat(food.get("timestamp", datetime.now().isoformat()))
                    try:
                        # Buscar eventos de phi 15-60 minutos após o alimento
                        # Isso seria implementado usando uma busca temporal no banco de dados real
                        phi_impact = food.get("metadata", {}).get("phi_impact", 0)
                        if phi_impact != 0:
                            phi_impacts.append(phi_impact)
                    except Exception as e:
                        logger.error(f"Erro ao analisar impacto de phi: {str(e)}")
                
                # Gerar recomendações baseadas na análise
                if phi_impacts:
                    avg_impact = sum(phi_impacts) / len(phi_impacts)
                    
                    if avg_impact < -0.05:  # Impacto negativo no phi
                        recommendations.append({
                            "type": "nutrition",
                            "name": "alerta nutricional",
                            "message": f"Este alimento tem histórico de redução média de phi em {abs(avg_impact):.2f}. Considere alternar com opções mais leves."
                        })
                    elif avg_impact > 0.05:  # Impacto positivo no phi
                        recommendations.append({
                            "type": "nutrition",
                            "name": "reforço positivo",
                            "message": f"Ótima escolha! Este alimento tem histórico de aumento médio de phi em {avg_impact:.2f}."
                        })
                
                # Recomendações baseadas no status atual
                if status == "low" and phi < 0.5:
                    recommendations.append({
                        "type": "nutrition",
                        "name": "sugestão para aumentar phi",
                        "message": "Seu phi está baixo. Considere alimentos leves, hidratação e uma pequena pausa após comer."
                    })
                elif status == "high" and phi > 0.9:
                    recommendations.append({
                        "type": "nutrition",
                        "name": "sugestão para estabilizar phi",
                        "message": "Seu phi está muito alto. Considere alimentos com carboidratos complexos para estabilização."
                    })
            
        # Recomendações genéricas
        recommendations.append({
            "type": "general",
            "name": "journaling",
            "message": "Anote três coisas positivas do seu dia para aumentar a resiliência ao estresse."
        })
        
        return recommendations
        
    async def listener_loop(self):
        """Loop de escuta de eventos de saúde."""
        self.is_listening = True
        
        try:
            logger.info("Iniciando escuta de eventos de saúde...")
            
            while self.is_listening:
                # Verificar alertas
                alerts = await self.check_for_alerts()
                
                # Processar alertas
                for alert in alerts:
                    # Gerar recomendações
                    recommendations = await self.get_recommendations(alert)
                    
                    # Armazenar alerta e recomendações
                    metadata = {
                        "alert": alert,
                        "recommendations": recommendations
                    }
                    
                    # Armazenar como evento de contexto
                    await self.store_health_event(
                        event_type="alert",
                        event_name=f"Alerta de frequência: {alert.get('event_type')}",
                        notes=alert.get("message", ""),
                        metadata=metadata
                    )
                    
                # Aguardar próximo ciclo (a cada 15 minutos)
                await asyncio.sleep(15 * 60)
                
        except asyncio.CancelledError:
            logger.info("Escuta de eventos de saúde interrompida")
            self.is_listening = False
        except Exception as e:
            logger.error(f"Erro no loop de escuta: {str(e)}")
            self.is_listening = False
            
    def start_auto_listening(self):
        """Inicia a escuta automática de eventos de saúde."""
        if self.listener_task is not None and not self.listener_task.done():
            logger.warning("Escuta automática já está em execução")
            return
            
        if not self.ready:
            logger.error("Sistema de memória não está pronto. Não é possível iniciar escuta automática")
            return
            
        # Criar e iniciar tarefa de escuta
        self.listener_task = asyncio.create_task(self.listener_loop())
        logger.info("Escuta automática iniciada")
        
    def stop_auto_listening(self):
        """Interrompe a escuta automática de eventos de saúde."""
        if self.listener_task is None or self.listener_task.done():
            logger.warning("Escuta automática não está em execução")
            return
            
        # Cancelar tarefa
        self.listener_task.cancel()
        self.is_listening = False
        logger.info("Escuta automática interrompida")

# Função para obter instância singleton
_health_hooks_instance = None

def get_health_hooks_instance(
    health_hooks_url: str = "http://localhost:5050/health-hooks",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    reinitialize: bool = False
) -> HealthHooksConnector:
    """
    Obtém ou cria instância singleton do conector Health-Hooks.
    
    Args:
        health_hooks_url: URL do serviço Health-Hooks
        qdrant_host: Host do Qdrant
        qdrant_port: Porta do Qdrant
        reinitialize: Forçar reinicialização da instância
        
    Returns:
        Instância do conector
    """
    global _health_hooks_instance
    
    if _health_hooks_instance is None or reinitialize:
        _health_hooks_instance = HealthHooksConnector(
            health_hooks_url=health_hooks_url,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port
        )
        
    return _health_hooks_instance