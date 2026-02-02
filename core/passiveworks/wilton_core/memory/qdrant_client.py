#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qdrant Vector Database Integration for WiltonOS

Este módulo implementa a memória vetorial de longo prazo usando o Qdrant
para armazenar e recuperar padrões de coerência, logs de exploração e
contextos relevantes para o balanceamento 3:1.

Principais recursos:
1. Armazenamento eficiente de medições de coerência (phi)
2. Busca por similaridade de estados de coerência anteriores
3. Persistência de dados para análise retrospectiva
4. Fornecimento de contexto para decisões de ajuste de modelo

Dependências: qdrant-client, numpy, openai
"""

import os
import time
import uuid
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Importar Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Configurar Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("qdrant_memory")

# Configurações padrão
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.environ.get("QDRANT_GRPC_PORT", "6334"))
VECTOR_SIZE = 1536  # Tamanho padrão para embeddings OpenAI
COHERENCE_COLLECTION = "coherence_memory"
EXPLORATION_COLLECTION = "exploration_memory"
CONTEXT_COLLECTION = "context_memory"

# Classe para gerenciar o Qdrant
class QdrantMemory:
    """
    Implementa memória vetorial de longo prazo utilizando o Qdrant como backend.
    
    Permite armazenar e recuperar estados de coerência, logs de exploração,
    e contextos relevantes para o balanceamento 3:1 do WiltonOS.
    """
    
    def __init__(self, 
                 host: str = QDRANT_HOST, 
                 port: int = QDRANT_PORT,
                 grpc_port: int = QDRANT_GRPC_PORT,
                 vector_size: int = VECTOR_SIZE,
                 openai_api_key: Optional[str] = None):
        """
        Inicializa a conexão com o Qdrant e configura as coleções necessárias.
        
        Args:
            host: Host do servidor Qdrant
            port: Porta HTTP do servidor Qdrant
            grpc_port: Porta gRPC do servidor Qdrant
            vector_size: Tamanho dos vetores de embedding
            openai_api_key: Chave de API do OpenAI (opcional, usa variável de ambiente se não fornecida)
        """
        self.vector_size = vector_size
        
        # Configurar OpenAI para geração de embeddings (caso necessário)
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        # Inicializar cliente Qdrant
        try:
            self.client = QdrantClient(host=host, port=port)
            self.ready = True
            logger.info(f"Conectado ao Qdrant em {host}:{port}")
        except Exception as e:
            logger.error(f"Erro ao conectar ao Qdrant: {str(e)}")
            self.ready = False
            
        # Inicializar as coleções necessárias se estiverem prontas
        if self.ready:
            self._initialize_collections()
            
    def _initialize_collections(self):
        """Inicializa as coleções necessárias para o funcionamento da memória."""
        collections = [
            (COHERENCE_COLLECTION, "Armazena estados e medições de coerência (phi)"),
            (EXPLORATION_COLLECTION, "Armazena logs de eventos de exploração"),
            (CONTEXT_COLLECTION, "Armazena contextos para tomada de decisão")
        ]
        
        for collection_name, description in collections:
            try:
                # Verificar se a coleção existe
                collection_info = self.client.get_collection(collection_name=collection_name)
                logger.info(f"Coleção {collection_name} já existe")
            except Exception:
                # Criar a coleção se não existir
                logger.info(f"Criando coleção {collection_name}...")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_size,
                        distance=qmodels.Distance.COSINE
                    )
                )
                # Nota: metadata não é suportado como parâmetro direto na criação da coleção
                logger.info(f"Coleção {collection_name} criada com sucesso")
                
    def health_check(self) -> bool:
        """
        Verifica se a conexão com o Qdrant está saudável.
        
        Returns:
            True se o Qdrant está acessível, False caso contrário
        """
        if not self.ready:
            return False
            
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            expected_collections = [COHERENCE_COLLECTION, EXPLORATION_COLLECTION, CONTEXT_COLLECTION]
            
            # Verificar se todas as coleções necessárias existem
            all_collections_exist = all(c in collection_names for c in expected_collections)
            
            if not all_collections_exist:
                logger.warning("Algumas coleções estão faltando. Inicializando...")
                self._initialize_collections()
                
            return True
        except Exception as e:
            logger.error(f"Erro no health check do Qdrant: {str(e)}")
            return False
            
    def get_embedding(self, text: str) -> List[float]:
        """
        Gera um embedding para o texto usando OpenAI.
        
        Args:
            text: Texto para gerar embedding
            
        Returns:
            Vetor de embedding
            
        Raises:
            ValueError: Se a API key do OpenAI não estiver disponível ou ocorrer um erro
        """
        if not self.openai_api_key:
            # Fallback para um embedding sintético (apenas para desenvolvimento)
            # Em produção, deve usar um modelo real de embedding
            logger.warning("API key do OpenAI não configurada. Usando embedding sintético.")
            rng = np.random.default_rng(seed=hash(text) % 2**32)
            embedding = rng.random(self.vector_size).astype(np.float32)
            # Normalizar o vetor
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.tolist()
            
        try:
            import openai
            openai.api_key = self.openai_api_key
            
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {str(e)}")
            raise ValueError(f"Falha ao gerar embedding via OpenAI: {str(e)}")
            
    def store_coherence_snapshot(self, 
                                 phi: float, 
                                 status: str,
                                 metadata: Dict[str, Any]) -> str:
        """
        Armazena um snapshot de coerência no Qdrant.
        
        Args:
            phi: Valor phi medido
            status: Status da coerência (ex: "low", "optimal", "high")
            metadata: Metadados adicionais sobre o estado
            
        Returns:
            ID do registro criado
            
        Raises:
            ConnectionError: Se não houver conexão com o Qdrant
        """
        if not self.ready or not self.health_check():
            raise ConnectionError("Qdrant não está disponível")
            
        # Criar descrição para embedding
        timestamp = datetime.now().isoformat()
        description = f"Phi: {phi:.4f}, Status: {status}, Time: {timestamp}"
        
        # Adicionar contexto dos metadados se disponível
        context_str = ""
        if "models" in metadata:
            models_info = metadata.get("models", {})
            model_names = list(models_info.keys())
            context_str += f" Models: {', '.join(model_names[:3])}..."
            
        if "weights" in metadata:
            weights = metadata.get("weights", {})
            weight_str = ", ".join([f"{k[:5]}:{v:.2f}" for k, v in list(weights.items())[:3]])
            context_str += f" Weights: {weight_str}..."
            
        if "recent_prompts" in metadata:
            prompts = metadata.get("recent_prompts", [])
            if prompts:
                # Truncar prompts longos
                prompt_summary = prompts[0][:50] + "..." if len(prompts[0]) > 50 else prompts[0]
                context_str += f" Recent: {prompt_summary}"
                
        # Combinar descrição e contexto
        embedding_text = description + context_str
        
        # Gerar embedding
        try:
            embedding = self.get_embedding(embedding_text)
        except ValueError:
            # Fallback para embedding simples baseado em phi
            # Isso é apenas um placeholder - em produção, use um modelo real
            logger.warning("Usando embedding de fallback baseado em phi")
            base_vector = np.zeros(self.vector_size)
            # Phi está entre 0-1, usamos para modificar os primeiros elementos do vetor
            base_vector[0] = phi 
            base_vector[1] = 1.0 if status == "optimal" else 0.5
            base_vector[2] = hash(timestamp) % 100 / 100.0
            # Adicionar aleatoriedade ao resto do vetor
            rng = np.random.default_rng(seed=int(time.time() * 1000))
            base_vector[3:] = rng.random(self.vector_size - 3)
            # Normalizar
            embedding = (base_vector / np.linalg.norm(base_vector)).tolist()
        
        # Criar ID único
        point_id = str(uuid.uuid4())
        
        # Preparar payload
        point = qmodels.PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "phi": phi,
                "status": status,
                "timestamp": timestamp,
                "description": description,
                "metadata": metadata
            }
        )
        
        # Armazenar no Qdrant
        self.client.upsert(
            collection_name=COHERENCE_COLLECTION,
            points=[point]
        )
        
        logger.info(f"Snapshot de coerência armazenado: {point_id} - Phi: {phi:.4f}")
        return point_id
        
    def find_similar_coherence_states(self, 
                                      phi: float, 
                                      status: str,
                                      metadata: Dict[str, Any] = None,
                                      limit: int = 5) -> List[Dict[str, Any]]:
        """
        Encontra estados de coerência similares ao estado atual.
        
        Args:
            phi: Valor phi atual
            status: Status da coerência atual
            metadata: Metadados adicionais sobre o estado atual
            limit: Número máximo de resultados a retornar
            
        Returns:
            Lista de estados de coerência similares
            
        Raises:
            ConnectionError: Se não houver conexão com o Qdrant
        """
        if not self.ready or not self.health_check():
            raise ConnectionError("Qdrant não está disponível")
            
        # Criar descrição para embedding de busca
        timestamp = datetime.now().isoformat()
        query_text = f"Phi: {phi:.4f}, Status: {status}, Time: {timestamp}"
        
        # Adicionar contexto dos metadados se disponível
        if metadata:
            if "models" in metadata:
                models_info = metadata.get("models", {})
                model_names = list(models_info.keys())
                query_text += f" Models: {', '.join(model_names[:3])}..."
                
            if "recent_prompts" in metadata:
                prompts = metadata.get("recent_prompts", [])
                if prompts:
                    # Truncar prompts longos
                    prompt_summary = prompts[0][:50] + "..." if len(prompts[0]) > 50 else prompts[0]
                    query_text += f" Recent: {prompt_summary}"
        
        # Gerar embedding para busca
        try:
            query_embedding = self.get_embedding(query_text)
        except ValueError:
            # Fallback para embedding simples baseado em phi
            logger.warning("Usando embedding de fallback baseado em phi para busca")
            base_vector = np.zeros(self.vector_size)
            base_vector[0] = phi 
            base_vector[1] = 1.0 if status == "optimal" else 0.5
            # Normalizar
            query_embedding = (base_vector / np.linalg.norm(base_vector)).tolist()
            
        # Buscar pontos similares
        search_result = self.client.search(
            collection_name=COHERENCE_COLLECTION,
            query_vector=query_embedding,
            limit=limit
        )
        
        # Formatar resultados
        results = []
        for point in search_result:
            results.append({
                "id": point.id,
                "phi": point.payload.get("phi"),
                "status": point.payload.get("status"),
                "timestamp": point.payload.get("timestamp"),
                "description": point.payload.get("description"),
                "metadata": point.payload.get("metadata"),
                "similarity": point.score
            })
            
        return results
        
    def store_exploration_event(self, 
                               event_type: str,
                               prompt: str,
                               model: str,
                               metadata: Dict[str, Any]) -> str:
        """
        Armazena um evento de exploração no Qdrant.
        
        Args:
            event_type: Tipo de evento de exploração
            prompt: Prompt utilizado
            model: Modelo utilizado
            metadata: Metadados adicionais
            
        Returns:
            ID do registro criado
        """
        if not self.ready or not self.health_check():
            raise ConnectionError("Qdrant não está disponível")
            
        # Criar descrição para embedding
        timestamp = datetime.now().isoformat()
        description = f"Event: {event_type}, Model: {model}, Time: {timestamp}"
        
        # Texto para embedding combina descrição e prompt
        embedding_text = description + " Prompt: " + prompt[:200]
        
        # Gerar embedding
        try:
            embedding = self.get_embedding(embedding_text)
        except ValueError:
            # Fallback
            logger.warning("Usando embedding de fallback para evento de exploração")
            rng = np.random.default_rng(seed=hash(embedding_text) % 2**32)
            embedding = rng.random(self.vector_size).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embedding = embedding.tolist()
            
        # Criar ID único
        point_id = str(uuid.uuid4())
        
        # Preparar payload
        point = qmodels.PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "event_type": event_type,
                "prompt": prompt,
                "model": model,
                "timestamp": timestamp,
                "metadata": metadata
            }
        )
        
        # Armazenar no Qdrant
        self.client.upsert(
            collection_name=EXPLORATION_COLLECTION,
            points=[point]
        )
        
        logger.info(f"Evento de exploração armazenado: {point_id} - Tipo: {event_type}")
        return point_id
        
    def find_similar_exploration_events(self, 
                                       prompt: str,
                                       event_type: Optional[str] = None,
                                       limit: int = 5) -> List[Dict[str, Any]]:
        """
        Encontra eventos de exploração similares com base no prompt.
        
        Args:
            prompt: Prompt para buscar eventos similares
            event_type: Filtrar por tipo de evento (opcional)
            limit: Número máximo de resultados
            
        Returns:
            Lista de eventos de exploração similares
        """
        if not self.ready or not self.health_check():
            raise ConnectionError("Qdrant não está disponível")
            
        # Texto para embedding
        query_text = prompt[:200]  # Limitar tamanho do prompt
        
        # Gerar embedding
        try:
            query_embedding = self.get_embedding(query_text)
        except ValueError:
            # Fallback
            logger.warning("Usando embedding de fallback para busca de eventos")
            rng = np.random.default_rng(seed=hash(query_text) % 2**32)
            query_embedding = rng.random(self.vector_size).astype(np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = query_embedding.tolist()
            
        # Filtro para tipo de evento (se especificado)
        filter_param = None
        if event_type:
            filter_param = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="event_type",
                        match=qmodels.MatchValue(value=event_type)
                    )
                ]
            )
            
        # Buscar pontos similares
        search_result = self.client.search(
            collection_name=EXPLORATION_COLLECTION,
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
                "prompt": point.payload.get("prompt"),
                "model": point.payload.get("model"),
                "timestamp": point.payload.get("timestamp"),
                "metadata": point.payload.get("metadata"),
                "similarity": point.score
            })
            
        return results
        
    def store_context(self,
                     context_type: str,
                     content: str,
                     metadata: Dict[str, Any]) -> str:
        """
        Armazena um contexto no Qdrant para uso futuro.
        
        Args:
            context_type: Tipo de contexto (ex: "user_feedback", "system_config")
            content: Conteúdo do contexto
            metadata: Metadados adicionais
            
        Returns:
            ID do registro criado
        """
        if not self.ready or not self.health_check():
            raise ConnectionError("Qdrant não está disponível")
            
        # Texto para embedding
        timestamp = datetime.now().isoformat()
        embedding_text = f"Type: {context_type}, Content: {content[:200]}"
        
        # Gerar embedding
        try:
            embedding = self.get_embedding(embedding_text)
        except ValueError:
            # Fallback
            logger.warning("Usando embedding de fallback para contexto")
            rng = np.random.default_rng(seed=hash(embedding_text) % 2**32)
            embedding = rng.random(self.vector_size).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embedding = embedding.tolist()
            
        # Criar ID único
        point_id = str(uuid.uuid4())
        
        # Preparar payload
        point = qmodels.PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "context_type": context_type,
                "content": content,
                "timestamp": timestamp,
                "metadata": metadata
            }
        )
        
        # Armazenar no Qdrant
        self.client.upsert(
            collection_name=CONTEXT_COLLECTION,
            points=[point]
        )
        
        logger.info(f"Contexto armazenado: {point_id} - Tipo: {context_type}")
        return point_id
        
    def find_relevant_context(self,
                             query: str,
                             context_type: Optional[str] = None,
                             limit: int = 5) -> List[Dict[str, Any]]:
        """
        Encontra contextos relevantes com base na consulta.
        
        Args:
            query: Texto da consulta
            context_type: Filtrar por tipo de contexto (opcional)
            limit: Número máximo de resultados
            
        Returns:
            Lista de contextos relevantes
        """
        if not self.ready or not self.health_check():
            raise ConnectionError("Qdrant não está disponível")
            
        # Gerar embedding
        try:
            query_embedding = self.get_embedding(query)
        except ValueError:
            # Fallback
            logger.warning("Usando embedding de fallback para busca de contextos")
            rng = np.random.default_rng(seed=hash(query) % 2**32)
            query_embedding = rng.random(self.vector_size).astype(np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = query_embedding.tolist()
            
        # Filtro para tipo de contexto (se especificado)
        filter_param = None
        if context_type:
            filter_param = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="context_type",
                        match=qmodels.MatchValue(value=context_type)
                    )
                ]
            )
            
        # Buscar pontos relevantes
        search_result = self.client.search(
            collection_name=CONTEXT_COLLECTION,
            query_vector=query_embedding,
            limit=limit,
            query_filter=filter_param
        )
        
        # Formatar resultados
        results = []
        for point in search_result:
            results.append({
                "id": point.id,
                "context_type": point.payload.get("context_type"),
                "content": point.payload.get("content"),
                "timestamp": point.payload.get("timestamp"),
                "metadata": point.payload.get("metadata"),
                "similarity": point.score
            })
            
        return results
        
    def get_collections_stats(self) -> Dict[str, Any]:
        """
        Obtém estatísticas sobre as coleções.
        
        Returns:
            Dicionário com estatísticas das coleções
        """
        if not self.ready or not self.health_check():
            raise ConnectionError("Qdrant não está disponível")
            
        stats = {}
        collections = [COHERENCE_COLLECTION, EXPLORATION_COLLECTION, CONTEXT_COLLECTION]
        
        for collection_name in collections:
            try:
                collection_info = self.client.get_collection(collection_name=collection_name)
                stats[collection_name] = {
                    "vectors_count": collection_info.vectors_count,
                    "points_count": collection_info.points_count,
                    "segments_count": collection_info.segments_count,
                    "status": "ready" if collection_info.status == "green" else "not_ready"
                }
            except Exception as e:
                logger.error(f"Erro ao obter estatísticas da coleção {collection_name}: {str(e)}")
                stats[collection_name] = {"status": "error", "error": str(e)}
                
        return stats
        
    def delete_collection(self, collection_name: str) -> bool:
        """
        Remove uma coleção completa (usar com cuidado).
        
        Args:
            collection_name: Nome da coleção a remover
            
        Returns:
            True se a operação for bem-sucedida, False caso contrário
        """
        if not self.ready:
            return False
            
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.warning(f"Coleção {collection_name} removida")
            return True
        except Exception as e:
            logger.error(f"Erro ao remover coleção {collection_name}: {str(e)}")
            return False
            
    def reset(self) -> bool:
        """
        Reinicia todas as coleções (remove e recria - usar com extremo cuidado).
        
        Returns:
            True se a operação for bem-sucedida, False caso contrário
        """
        if not self.ready:
            return False
            
        try:
            collections = [COHERENCE_COLLECTION, EXPLORATION_COLLECTION, CONTEXT_COLLECTION]
            
            for collection_name in collections:
                try:
                    self.client.delete_collection(collection_name=collection_name)
                    logger.warning(f"Coleção {collection_name} removida")
                except Exception:
                    # Ignorar erro se a coleção não existir
                    pass
                    
            # Recriar coleções
            self._initialize_collections()
            return True
        except Exception as e:
            logger.error(f"Erro ao reiniciar coleções: {str(e)}")
            return False
            
# Função para obter uma instância singleton do QdrantMemory
_memory_instance = None

def get_memory_instance(
    host: str = QDRANT_HOST,
    port: int = QDRANT_PORT,
    reinitialize: bool = False
) -> QdrantMemory:
    """
    Obtém ou cria a instância singleton de QdrantMemory.
    
    Args:
        host: Host do servidor Qdrant
        port: Porta HTTP do servidor Qdrant
        reinitialize: Forçar reinicialização da instância
        
    Returns:
        Instância de QdrantMemory
    """
    global _memory_instance
    
    if _memory_instance is None or reinitialize:
        _memory_instance = QdrantMemory(host=host, port=port)
        
    return _memory_instance

# Função para obter cliente direto do Qdrant
_qdrant_client_instance = None

def get_qdrant_client(
    host: str = QDRANT_HOST,
    port: int = QDRANT_PORT,
    reinitialize: bool = False
) -> QdrantClient:
    """
    Obtém ou cria uma instância singleton do cliente Qdrant.
    
    Args:
        host: Host do servidor Qdrant
        port: Porta HTTP do servidor Qdrant
        reinitialize: Forçar reinicialização da instância
        
    Returns:
        Instância de QdrantClient
    """
    global _qdrant_client_instance
    
    if _qdrant_client_instance is None or reinitialize:
        try:
            _qdrant_client_instance = QdrantClient(host=host, port=port)
            logger.info(f"Cliente Qdrant inicializado em {host}:{port}")
        except Exception as e:
            logger.error(f"Erro ao conectar ao Qdrant: {str(e)}")
            raise ConnectionError(f"Não foi possível conectar ao Qdrant: {str(e)}")
        
    return _qdrant_client_instance