#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoPoster para WiltonOS

Este módulo implementa um sistema de geração automatizada de publicações 
baseado no estado de coerência (phi) do sistema.

Características principais:
1. Geração de sugestões de postagem com base no phi atual
2. Adaptação do tom e conteúdo ao equilíbrio quântico do sistema
3. Integração com a memória vetorial para continuidade temática
4. Otimização para o balanceamento 3:1 (75% coerência, 25% exploração)
5. Calibração fractal para maximizar ressonância com audiências específicas

Dependências: qdrant-client, openai
"""

import os
import json
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("auto_poster")

# Importar módulos do WiltonOS
from wilton_core.memory.health_hooks_connector import get_health_hooks_instance
from wilton_core.memory.qdrant_client import get_memory_instance

# Configurações
POST_COLLECTION = "post_suggestions"
POST_HISTORY_LIMIT = 20
DEFAULT_TARGET_PHI = 0.75
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Estados quânticos e seus atributos para posts
QUANTUM_STATES = {
    "exploração_excessiva": {
        "description": "Estado de exploração excessiva (phi < 0.25)",
        "traits": ["abstrato", "experimental", "provocativo", "disruptivo", "caótico"],
        "tone": "especulativo e exploratório",
        "style": "não-linear e associativo"
    },
    "transição": {
        "description": "Estado de transição (0.25 <= phi < 0.65)",
        "traits": ["balanceado", "transitório", "emergente", "conectivo", "integrativo"],
        "tone": "reflexivo e inquisitivo",
        "style": "conectando ideias diversas"
    },
    "sweet_spot": {
        "description": "Estado ótimo (0.65 <= phi <= 0.85)",
        "traits": ["preciso", "estruturado", "profundo", "sistemático", "coerente"],
        "tone": "assertivo e profundo",
        "style": "sistêmico e integrado"
    },
    "supercoerência": {
        "description": "Estado de supercoerência (phi > 0.85)",
        "traits": ["meticuloso", "refinado", "consolidado", "dogmático", "detalhista"],
        "tone": "autoritativo e definitivo",
        "style": "detalhado e convergente"
    }
}

# Categorias de posts
POST_CATEGORIES = [
    "insights_técnicos",
    "reflexões_filosóficas",
    "aplicações_práticas",
    "análises_sistêmicas",
    "experimentos_conceituais",
    "tendências_emergentes",
    "cases_complexos",
    "recursos_metodológicos"
]

class AutoPoster:
    """
    Sistema de geração automatizada de publicações baseado no estado de coerência phi.
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 auto_initialize: bool = True):
        """
        Inicializa o sistema AutoPoster.
        
        Args:
            openai_api_key: Chave da API OpenAI
            auto_initialize: Inicializar automaticamente conexões
        """
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.ready = False
        self.health_connector = None
        self.memory = None
        
        if auto_initialize:
            self.initialize()
            
    def initialize(self) -> bool:
        """
        Inicializa conexões com health hooks e memória.
        
        Returns:
            True se inicializado com sucesso, False caso contrário
        """
        try:
            # Conectar ao Health Hooks para acessar phi
            self.health_connector = get_health_hooks_instance()
            
            # Conectar à memória vetorial
            self.memory = get_memory_instance()
            
            # Verificar se a coleção de posts existe
            self._initialize_collection()
            
            self.ready = True
            logger.info("Sistema AutoPoster inicializado com sucesso")
            return True
        except Exception as e:
            logger.error(f"Erro ao inicializar AutoPoster: {str(e)}")
            self.ready = False
            return False
            
    def _initialize_collection(self):
        """Inicializa a coleção para sugestões de posts."""
        try:
            if not self.memory or not hasattr(self.memory, 'client'):
                logger.error("Cliente de memória não inicializado")
                return
                
            # Verificar se a coleção existe
            collections = self.memory.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if POST_COLLECTION not in collection_names:
                # Criar a coleção
                logger.info(f"Criando coleção {POST_COLLECTION}...")
                self.memory.client.create_collection(
                    collection_name=POST_COLLECTION,
                    vectors_config=self.memory.client.http.models.VectorParams(
                        size=self.memory.vector_size,
                        distance=self.memory.client.http.models.Distance.COSINE
                    )
                )
                logger.info(f"Coleção {POST_COLLECTION} criada com sucesso")
            else:
                logger.info(f"Coleção {POST_COLLECTION} já existe")
        except Exception as e:
            logger.error(f"Erro ao inicializar coleção de posts: {str(e)}")
            
    async def get_coherence_state(self) -> Dict[str, Any]:
        """
        Obtém o estado atual de coerência do sistema.
        
        Returns:
            Um dicionário contendo o phi atual, o desvio, o alinhamento fractal
            e o estado quântico
        """
        if not self.ready or not self.health_connector:
            logger.warning("Sistema não está pronto para obter estado de coerência")
            return {
                "phi": 0.5,
                "deviation": -0.25,
                "alignment": 0.67,
                "state": "transição"
            }
            
        try:
            coherence_state = await self.health_connector.get_coherence_state()
            phi = coherence_state.get("phi", 0.5)
            
            # Calcular métricas derivadas
            deviation = phi - DEFAULT_TARGET_PHI
            alignment = 1.0 - min(abs(deviation) / DEFAULT_TARGET_PHI, 1.0)
            
            # Determinar estado quântico
            quantum_state = "exploração_excessiva" if phi < 0.25 else \
                           "supercoerência" if phi > 0.85 else \
                           "sweet_spot" if abs(phi - DEFAULT_TARGET_PHI) <= 0.1 else "transição"
                           
            return {
                "phi": phi,
                "deviation": deviation,
                "alignment": alignment,
                "state": quantum_state
            }
        except Exception as e:
            logger.error(f"Erro ao obter estado de coerência: {str(e)}")
            return {
                "phi": 0.5,
                "deviation": -0.25,
                "alignment": 0.67,
                "state": "transição"
            }
            
    async def generate_post_suggestion(self,
                                     target_phi: float = DEFAULT_TARGET_PHI,
                                     categories: Optional[List[str]] = None,
                                     context: Optional[str] = None) -> Dict[str, Any]:
        """
        Gera uma sugestão de publicação com base no phi atual.
        
        Args:
            target_phi: Valor phi alvo para calibrar a sugestão
            categories: Lista de categorias para filtrar sugestões
            context: Contexto adicional para informar a geração
            
        Returns:
            Sugestão de publicação formatada
        """
        if not self.ready:
            logger.warning("Sistema não está completamente inicializado. Usando modo fallback.")
            # Gerar post sintético mesmo sem o sistema estar completamente inicializado
            # para garantir que a funcionalidade básica da API esteja sempre disponível
            coherence_state = {
                "phi": 0.5,
                "deviation": -0.25,
                "alignment": 0.67,
                "state": "transição"
            }
            # Usar a função existente para posts sintéticos
            return self._generate_synthetic_post(
                coherence_state["state"],
                categories if categories else ["insights_técnicos", "reflexões_filosóficas"],
                ["balanceado", "transitório", "emergente"]
            )
            
        try:
            # Obter estado atual de coerência
            coherence_state = await self.get_coherence_state()
            current_phi = coherence_state["phi"]
            quantum_state = coherence_state["state"]
            
            # Diferença entre phi atual e alvo
            phi_gap = target_phi - current_phi
            
            # Calibrar o tipo de publicação com base no gap
            calibration_factor = min(abs(phi_gap), 0.5) * 2  # 0-1 scale
            
            # Escolher categorias se não fornecidas
            if not categories or len(categories) == 0:
                # Em phi baixo, favorecer categorias mais exploratórias
                if current_phi < 0.4:
                    category_weights = {
                        "experimentos_conceituais": 0.3,
                        "tendências_emergentes": 0.2,
                        "reflexões_filosóficas": 0.2,
                        "insights_técnicos": 0.1,
                        "aplicações_práticas": 0.1,
                        "análises_sistêmicas": 0.05,
                        "cases_complexos": 0.03,
                        "recursos_metodológicos": 0.02
                    }
                # Em phi alto, favorecer categorias mais estruturadas
                elif current_phi > 0.6:
                    category_weights = {
                        "insights_técnicos": 0.25,
                        "análises_sistêmicas": 0.2,
                        "aplicações_práticas": 0.2,
                        "recursos_metodológicos": 0.15,
                        "cases_complexos": 0.1,
                        "reflexões_filosóficas": 0.05,
                        "tendências_emergentes": 0.03,
                        "experimentos_conceituais": 0.02
                    }
                # Em phi médio, equilibrar
                else:
                    category_weights = {cat: 1.0/len(POST_CATEGORIES) for cat in POST_CATEGORIES}
                    
                # Escolher categorias com base nos pesos
                selected_categories = random.choices(
                    list(category_weights.keys()),
                    weights=list(category_weights.values()),
                    k=min(3, len(category_weights))
                )
            else:
                selected_categories = categories[:3]  # Limitar a 3 categorias
                
            # Obter características do estado quântico atual
            state_traits = QUANTUM_STATES.get(quantum_state, QUANTUM_STATES["transição"])
            
            # Construir prompt para geração do post
            selected_traits = random.sample(state_traits["traits"], min(3, len(state_traits["traits"])))
            
            # Obter histórico recente para continuidade
            recent_posts = await self._get_recent_posts(limit=3)
            recent_titles = [p.get("title", "") for p in recent_posts if "title" in p]
            recent_context = "\n".join([f"- {title}" for title in recent_titles])
            
            # Construir instrução baseada no gap phi
            if phi_gap < -0.2:  # Muito coerente, precisa explorar mais
                instruction = "Crie uma publicação mais exploratória e criativa que desafie paradigmas estabelecidos."
            elif phi_gap > 0.2:  # Muito exploratório, precisa consolidar
                instruction = "Crie uma publicação mais estruturada e integrativa que consolide aprendizados existentes."
            else:  # Próximo do equilíbrio ideal
                instruction = "Crie uma publicação balanceada que conecte ideias em um framework coerente, mantendo espaço para exploração."
                
            # Gerar ideia de publicação usando OpenAI ou similar
            post_suggestion = await self._generate_post_with_ai(
                quantum_state=quantum_state,
                state_traits=state_traits,
                selected_categories=selected_categories,
                selected_traits=selected_traits,
                recent_context=recent_context,
                instruction=instruction,
                context=context
            )
            
            # Adicionar metadados de phi
            post_suggestion["phi_context"] = {
                "current_phi": current_phi,
                "target_phi": target_phi,
                "phi_gap": phi_gap,
                "quantum_state": quantum_state,
                "calibration_factor": calibration_factor
            }
            
            # Armazenar sugestão para uso futuro
            await self._store_post_suggestion(post_suggestion)
            
            logger.info(f"Sugestão de post gerada: {post_suggestion.get('title', 'Sem título')}")
            return post_suggestion
        except Exception as e:
            logger.error(f"Erro ao gerar sugestão de publicação: {str(e)}")
            return {
                "error": f"Falha ao gerar sugestão: {str(e)}",
                "fallback_suggestion": {
                    "title": "Balanceando Exploração e Coerência em Sistemas Complexos",
                    "summary": "Uma análise do equilíbrio 3:1 entre coerência estruturada e exploração dinâmica.",
                    "topics": ["equilíbrio quântico", "coerência phi", "exploração fractal"]
                }
            }
            
    async def _generate_post_with_ai(self,
                                   quantum_state: str,
                                   state_traits: Dict[str, Any],
                                   selected_categories: List[str],
                                   selected_traits: List[str],
                                   recent_context: str,
                                   instruction: str,
                                   context: Optional[str] = None) -> Dict[str, Any]:
        """
        Gera uma sugestão de post usando a API OpenAI.
        
        Args:
            quantum_state: Estado quântico atual
            state_traits: Características do estado quântico
            selected_categories: Categorias selecionadas
            selected_traits: Traços selecionados para o post
            recent_context: Contexto de posts recentes
            instruction: Instrução de calibração
            context: Contexto adicional
            
        Returns:
            Sugestão de publicação formatada
        """
        import json
        
        if not self.openai_api_key:
            return self._generate_synthetic_post(
                quantum_state, 
                selected_categories, 
                selected_traits
            )
            
        try:
            import openai
            openai.api_key = self.openai_api_key
            
            # Construir o prompt
            system_prompt = f"""
            Você é um gerador de ideias para publicações (posts) de alta qualidade no contexto do WiltonOS.
            Gere uma sugestão de publicação com base no estado quântico atual do sistema: {quantum_state} ({state_traits['description']}).

            O post deve ter as seguintes características:
            - Categorias: {', '.join(selected_categories)}
            - Traços de estilo: {', '.join(selected_traits)}
            - Tom: {state_traits['tone']}
            - Estilo: {state_traits['style']}

            Posts recentes (para continuidade temática):
            {recent_context}

            {instruction}

            {context if context else ""}

            Responda em formato JSON com os seguintes campos:
            - title: título chamativo para o post
            - summary: resumo conciso do conteúdo (máximo 2 parágrafos)
            - topics: lista de 3-5 tópicos-chave abordados
            - categories: lista das categorias fornecidas
            - key_points: 3-5 pontos principais que o post deve cobrir
            - content_structure: estrutura sugerida para o conteúdo (introdução, seções principais, conclusão)
            - tone: tom recomendado para o post
            - estimated_length: comprimento estimado em palavras
            """
            
            response = openai.chat.completions.create(
                model="gpt-4o", # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": system_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=1000,
            )
            
            # Extrair e validar o JSON
            suggestion = json.loads(response.choices[0].message.content)
            
            # Adicionar timestamp e ID
            suggestion["generated_at"] = datetime.now().isoformat()
            suggestion["id"] = str(uuid.uuid4())
            suggestion["quantum_state"] = quantum_state
            
            return suggestion
        except Exception as e:
            logger.error(f"Erro ao gerar post com IA: {str(e)}")
            return self._generate_synthetic_post(
                quantum_state, 
                selected_categories, 
                selected_traits
            )
            
    def _generate_synthetic_post(self,
                               quantum_state: str,
                               categories: List[str],
                               traits: List[str]) -> Dict[str, Any]:
        """
        Gera uma sugestão de post sintética quando a IA não está disponível.
        
        Args:
            quantum_state: Estado quântico atual
            categories: Categorias selecionadas
            traits: Traços selecionados
            
        Returns:
            Sugestão de publicação sintética
        """
        # Templates de título por estado
        title_templates = {
            "exploração_excessiva": [
                "Explorando as Fronteiras de {topic} em {context}",
                "O Paradoxo de {topic}: Reimaginando {context}",
                "Além dos Limites: {topic} como {context} Emergente"
            ],
            "transição": [
                "Conectando {topic} com {context}: Uma Perspectiva Integrada",
                "A Evolução do {topic} para um Novo {context}",
                "Transformação de {topic}: Navegando a Transição para {context}"
            ],
            "sweet_spot": [
                "O Equilíbrio Ótimo entre {topic} e {context}",
                "Framework 3:1 para {topic} em Sistemas {context}",
                "Sustentando Coerência em {topic}: Estratégias {context}"
            ],
            "supercoerência": [
                "Análise Detalhada de {topic} no Contexto {context}",
                "Refinamento Sistemático de {topic} para Máxima {context}",
                "Padrões Consolidados de {topic}: Implicações para {context}"
            ]
        }
        
        # Tópicos e contextos por categoria
        topics_contexts = {
            "insights_técnicos": (
                ["Algoritmos Adaptativos", "Arquitetura Fractal", "Computação Quântica", "Memória Vetorial"],
                ["Sistemas Dinâmicos", "Aprendizado de Máquina", "Processamento Distribuído", "Neurociência Computacional"]
            ),
            "reflexões_filosóficas": (
                ["Consciência Emergente", "Dualidade Quântica", "Paradoxos Sistêmicos", "Meta-Cognição"],
                ["Realidades Complexas", "Estudos de Percepção", "Epistemologia Fractal", "Fenomenologia"]
            ),
            "aplicações_práticas": (
                ["Auto-Calibração", "Biofeedback Integrado", "Análise Preditiva", "Optimização de Fluxos"],
                ["Saúde Pessoal", "Produtividade Cognitiva", "Desenvolvimento de Software", "Tomada de Decisão"]
            ),
            "análises_sistêmicas": (
                ["Padrões Recursivos", "Dinâmicas de Rede", "Ciclos de Feedback", "Emergência Estruturada"],
                ["Ecossistemas Digitais", "Organizações Complexas", "Evolução de Mercados", "Cognição Aumentada"]
            ),
            "experimentos_conceituais": (
                ["Interfaceamento Quântico", "Hibridização Cognitiva", "Topologia de Consciência", "Auto-Referenciação"],
                ["Mente Estendida", "Percepção Aumentada", "Identidade Digital", "Processos Criativos"]
            )
        }
        
        # Escolher categoria, tópico e contexto
        category = random.choice(categories if categories else list(topics_contexts.keys()))
        topics_list, contexts_list = topics_contexts.get(
            category, 
            (["Sistemas Complexos", "Integração Fractal"], ["Computação Moderna", "Arquiteturas Cognitivas"])
        )
        
        topic = random.choice(topics_list)
        context = random.choice(contexts_list)
        
        # Escolher template de título
        templates = title_templates.get(quantum_state, title_templates["transição"])
        title_template = random.choice(templates)
        title = title_template.format(topic=topic, context=context)
        
        # Gerar sumário
        summary = f"Uma exploração de {topic} no contexto de {context}, destacando padrões {random.choice(traits)} e abordagens {random.choice(traits)}. Este post aborda o equilíbrio entre estrutura e exploração, fornecendo insights para aplicação em sistemas complexos."
        
        # Gerar tópicos
        topics = [topic, context, f"{topic} em {context}", "Equilíbrio 3:1", "Padrões Fractais"]
        
        # Gerar estrutura
        structure = {
            "introdução": f"Contextualização de {topic} e sua relevância",
            "seção_1": f"Fundamentos teóricos de {topic}",
            "seção_2": f"Aplicações em {context}",
            "seção_3": f"Equilíbrio 3:1 para otimização",
            "conclusão": "Síntese e próximos passos"
        }
        
        # Montagem da sugestão
        suggestion = {
            "id": str(uuid.uuid4()),
            "title": title,
            "summary": summary,
            "topics": topics,
            "categories": [category],
            "key_points": [
                f"Fundamentos de {topic} em sistemas complexos",
                f"Integração com {context} para resultados otimizados",
                "Aplicação do equilíbrio quântico 3:1",
                f"Padrões de auto-organização em {topic}"
            ],
            "content_structure": structure,
            "tone": QUANTUM_STATES[quantum_state]["tone"],
            "estimated_length": random.choice([800, 1200, 1500, 2000]),
            "generated_at": datetime.now().isoformat(),
            "quantum_state": quantum_state,
            "synthetic": True
        }
        
        return suggestion
            
    async def _store_post_suggestion(self, post: Dict[str, Any]) -> Optional[str]:
        """
        Armazena uma sugestão de post no Qdrant.
        
        Args:
            post: Sugestão de post para armazenar
            
        Returns:
            ID do registro ou None em caso de erro
        """
        if not self.ready or not self.memory:
            logger.warning("Sistema não está pronto para armazenar sugestões")
            return None
            
        # Extrair texto para embedding
        post_text = f"{post.get('title', '')} - {post.get('summary', '')}"
        if 'topics' in post:
            topics_text = ", ".join(post['topics'])
            post_text += f" Topics: {topics_text}"
            
        # Gerar embedding
        try:
            embedding = self.memory.get_embedding(post_text)
        except ValueError:
            logger.warning("Erro ao gerar embedding. Usando embedding sintético")
            import numpy as np
            rng = np.random.default_rng(seed=hash(post_text) % 2**32)
            embedding = rng.random(self.memory.vector_size).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embedding = embedding.tolist()
            
        # Preparar ID
        point_id = post.get("id", str(uuid.uuid4()))
        
        # Preparar payload
        point = self.memory.client.http.models.PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "post": post,
                "text": post_text,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Armazenar no Qdrant
        try:
            self.memory.client.upsert(
                collection_name=POST_COLLECTION,
                points=[point]
            )
            logger.info(f"Sugestão de post armazenada: {point_id}")
            return point_id
        except Exception as e:
            logger.error(f"Erro ao armazenar sugestão de post: {str(e)}")
            return None
            
    async def _get_recent_posts(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Obtém sugestões de posts recentes.
        
        Args:
            limit: Número máximo de posts a retornar
            
        Returns:
            Lista de posts
        """
        if not self.ready or not self.memory:
            logger.warning("Sistema não está pronto para buscar posts recentes")
            return []
            
        try:
            # Obter registros mais recentes
            # Qdrant não tem ordenação por timestamp nativa, então precisamos fazer uma busca geral
            # e ordenar manualmente
            
            # Buscar com um vetor aleatório para obter todos os registros
            import numpy as np
            random_vector = np.random.random(self.memory.vector_size).astype(np.float32)
            random_vector = random_vector / np.linalg.norm(random_vector)
            
            # Buscar com limite maior para garantir que temos registros suficientes para ordenar
            search_limit = min(limit * 3, 100)
            search_result = self.memory.client.search(
                collection_name=POST_COLLECTION,
                query_vector=random_vector.tolist(),
                limit=search_limit
            )
            
            # Extrair posts e ordenar por timestamp
            posts = []
            for point in search_result:
                if "post" in point.payload:
                    post = point.payload["post"]
                    timestamp = point.payload.get("timestamp", "")
                    posts.append((post, timestamp))
            
            # Ordenar por timestamp (mais recente primeiro)
            posts.sort(key=lambda x: x[1], reverse=True)
            
            # Retornar apenas os posts limitados
            return [post for post, _ in posts[:limit]]
        except Exception as e:
            logger.error(f"Erro ao buscar posts recentes: {str(e)}")
            return []
            
    async def find_similar_posts(self, 
                               query: str,
                               limit: int = 5) -> List[Dict[str, Any]]:
        """
        Encontra sugestões de posts similares com base em uma consulta.
        
        Args:
            query: Texto da consulta
            limit: Número máximo de resultados
            
        Returns:
            Lista de posts similares
        """
        if not self.ready or not self.memory:
            logger.warning("Sistema não está pronto para buscar posts similares")
            return []
            
        # Gerar embedding para a consulta
        try:
            query_embedding = self.memory.get_embedding(query)
        except ValueError:
            logger.warning("Erro ao gerar embedding para consulta. Usando fallback")
            import numpy as np
            rng = np.random.default_rng(seed=hash(query) % 2**32)
            query_embedding = rng.random(self.memory.vector_size).astype(np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = query_embedding.tolist()
            
        # Buscar posts similares
        try:
            search_result = self.memory.client.search(
                collection_name=POST_COLLECTION,
                query_vector=query_embedding,
                limit=limit
            )
            
            # Formatar resultados
            results = []
            for point in search_result:
                if "post" in point.payload:
                    post = point.payload["post"]
                    post["similarity"] = point.score
                    results.append(post)
                    
            return results
        except Exception as e:
            logger.error(f"Erro ao buscar posts similares: {str(e)}")
            return []
            
    async def calibrate_for_phi_balance(self, current_phi: float) -> Dict[str, Any]:
        """
        Calibra recomendações de posts para otimizar o equilíbrio phi.
        
        Args:
            current_phi: Valor phi atual
            
        Returns:
            Recomendações calibradas
        """
        # Calcular gap e direção de calibração
        phi_gap = DEFAULT_TARGET_PHI - current_phi
        
        calibration = {
            "current_phi": current_phi,
            "target_phi": DEFAULT_TARGET_PHI,
            "phi_gap": phi_gap,
            "direction": "increase" if phi_gap > 0 else "decrease" if phi_gap < 0 else "maintain",
            "intensity": min(abs(phi_gap) * 2, 1.0),  # 0-1 scale
            "recommendations": {}
        }
        
        # Calibrar recomendações com base no gap
        if phi_gap > 0.2:  # Muito exploratório, precisa aumentar coerência
            calibration["recommendations"] = {
                "post_frequency": "aumentar",
                "post_structure": "mais estruturado e sistemático",
                "topics": ["frameworks integrativos", "consolidação de conhecimento", "padrões estabelecidos"],
                "recommended_categories": ["insights_técnicos", "análises_sistêmicas", "recursos_metodológicos"]
            }
        elif phi_gap < -0.2:  # Muito coerente, precisa aumentar exploração
            calibration["recommendations"] = {
                "post_frequency": "reduzir",
                "post_structure": "mais exploratório e experimental",
                "topics": ["ideias emergentes", "conexões não-óbvias", "questionamentos de paradigmas"],
                "recommended_categories": ["experimentos_conceituais", "reflexões_filosóficas", "tendências_emergentes"]
            }
        else:  # Próximo do equilíbrio ideal
            calibration["recommendations"] = {
                "post_frequency": "manter",
                "post_structure": "equilibrado entre estrutura e exploração",
                "topics": ["integração de ideias", "conexões entre domínios", "padrões fractais"],
                "recommended_categories": ["aplicações_práticas", "análises_sistêmicas", "insights_técnicos"]
            }
            
        return calibration
            

# Função para obter ou criar instância do AutoPoster
_AUTO_POSTER_INSTANCE = None

def get_auto_poster_instance(openai_api_key: Optional[str] = None,
                           reinitialize: bool = False) -> AutoPoster:
    """
    Obtém ou cria uma instância do AutoPoster.
    
    Args:
        openai_api_key: Chave da API OpenAI
        reinitialize: Forçar reinicialização da instância
        
    Returns:
        Instância do AutoPoster
    """
    global _AUTO_POSTER_INSTANCE
    
    if _AUTO_POSTER_INSTANCE is None or reinitialize:
        _AUTO_POSTER_INSTANCE = AutoPoster(openai_api_key=openai_api_key)
        
    return _AUTO_POSTER_INSTANCE