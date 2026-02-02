#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Food-Logging para WiltonOS

Este módulo implementa um sistema avançado de registro alimentar que utiliza
a memória vetorial para analisar padrões de alimentação e seus impactos no
nível de coerência (phi) do usuário.

Recursos principais:
1. Registro detalhado de alimentos consumidos
2. Classificação automática de ingredientes e categorias
3. Análise de impacto dos alimentos sobre phi no curto prazo (15-60 min)
4. Recomendações personalizadas baseadas em histórico e contexto
5. Indexação de alto desempenho usando embeddings de texto e ingredientes

O sistema permite criar um ciclo de feedback onde o usuário pode entender
como sua alimentação afeta seu nível de coerência e receber sugestões
para otimizar seu equilíbrio.
"""

import os
import sys
import time
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set

# Importar conector de memória
from .health_hooks_connector import get_health_hooks_instance

# Importar modelo fractal de nutrição
from wilton_core.models.fractal_phi_nutrition import FractalPhiNutrition, IDEAL_PHI

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/food_logging.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("food_logging")

# Categorias de alimentos
FOOD_CATEGORIES = {
    "proteínas": [
        "carne", "frango", "peixe", "ovo", "leguminosa", "feijão", "lentilha", 
        "grão de bico", "soja", "tofu", "queijo", "iogurte", "leite", "whey"
    ],
    "carboidratos": [
        "arroz", "pão", "massa", "macarrão", "batata", "mandioca", "milho", 
        "aveia", "quinoa", "cuscuz", "tapioca", "cereal"
    ],
    "gorduras": [
        "azeite", "óleo", "manteiga", "ghee", "óleo de coco", "abacate", 
        "castanha", "amêndoa", "noz", "semente", "tahine"
    ],
    "vegetais": [
        "alface", "espinafre", "brócolis", "couve", "rúcula", "tomate", 
        "cenoura", "cebola", "alho", "pimentão", "abobrinha", "berinjela", 
        "pepino", "repolho"
    ],
    "frutas": [
        "maçã", "banana", "uva", "laranja", "mexerica", "melancia", "mamão", 
        "morango", "abacaxi", "manga", "kiwi", "pêra", "melão", "açaí"
    ],
    "condimentos": [
        "sal", "pimenta", "orégano", "coentro", "manjericão", "salsa", 
        "cúrcuma", "gengibre", "canela", "cominho", "páprica", "curry"
    ],
    "bebidas": [
        "água", "café", "chá", "suco", "refrigerante", "kombucha", "cerveja", 
        "vinho", "bebida alcoólica"
    ],
    "processados": [
        "fast food", "industrializado", "ultraprocessado", "embutido", 
        "biscoito", "bolacha", "salgadinho", "fritura", "doce", 
        "suco de caixinha", "molho pronto"
    ]
}

class FoodLogger:
    """
    Implementa o sistema de registro alimentar para o WiltonOS.
    
    Oferece funções para registrar alimentos, analisar padrões e
    impactos na coerência, e gerar recomendações personalizadas.
    """
    
    def __init__(self,
                 health_hooks_url: str = "http://localhost:5050/health-hooks",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333):
        """
        Inicializa o registro alimentar.
        
        Args:
            health_hooks_url: URL do serviço Health-Hooks
            qdrant_host: Host do Qdrant
            qdrant_port: Porta do Qdrant
        """
        self.health_hooks_url = health_hooks_url
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        
        # Inicializar conector Health-Hooks
        try:
            self.health_connector = get_health_hooks_instance(
                health_hooks_url=health_hooks_url,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port
            )
            self.ready = self.health_connector.ready
            
            if self.ready:
                logger.info(f"Conectado ao sistema de memória Health-Hooks")
            else:
                logger.error(f"Falha ao conectar ao sistema Health-Hooks")
        except Exception as e:
            logger.error(f"Erro ao inicializar conector Health-Hooks: {str(e)}")
            self.health_connector = None
            self.ready = False
    
    async def register_food(self, 
                          food_name: str, 
                          ingredients: Optional[List[str]] = None,
                          notes: str = "",
                          meal_type: str = "refeição",
                          meal_time: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Registra um alimento consumido.
        
        Args:
            food_name: Nome do alimento ou refeição
            ingredients: Lista de ingredientes (opcional)
            notes: Observações sobre a refeição
            meal_type: Tipo de refeição (café da manhã, almoço, jantar, lanche)
            meal_time: Horário da refeição (formato ISO ou descrição)
            
        Returns:
            Informações do registro ou None em caso de erro
        """
        if not self.ready:
            logger.warning("Sistema de registro alimentar não está pronto")
            return None
            
        # Preparar metadados
        timestamp = datetime.now().isoformat()
        
        if meal_time is None:
            meal_time = timestamp
            
        # Extrair phi atual como linha de base para análise de impacto
        coherence_state = await self.health_connector.get_coherence_state()
        baseline_phi = coherence_state.get("phi", 0) if coherence_state else 0
        
        # Identificar categorias de alimentos
        if ingredients is None:
            # Se não fornecido, tentar extrair do nome
            ingredients = self._extract_ingredients(food_name)
            
        categories = self._categorize_food(ingredients)
        
        # Buscar alimentos similares
        similar_foods = await self.health_connector.find_similar_health_events(
            query=food_name,
            event_type="food",
            limit=3
        )
        
        # Analisar impactos históricos em phi
        phi_impacts = []
        for food in similar_foods:
            impact = food.get("metadata", {}).get("phi_impact", 0)
            if impact != 0:
                phi_impacts.append(impact)
                
        avg_impact = sum(phi_impacts) / len(phi_impacts) if phi_impacts else 0
        
        # Montar metadados completos
        metadata = {
            "food_name": food_name,
            "ingredients": ingredients,
            "categories": list(categories),
            "meal_type": meal_type,
            "meal_time": meal_time,
            "baseline_phi": baseline_phi,
            "similar_foods": [f.get("food_name", "") for f in similar_foods],
            "predicted_phi_impact": avg_impact,
            "actual_phi_impact": None,  # Será atualizado após medição posterior
            "timestamp": timestamp
        }
        
        # Registrar evento de alimentação
        try:
            event_id = await self.health_connector.store_health_event(
                event_type="food",
                event_name=food_name,
                notes=notes,
                metadata=metadata
            )
            
            if event_id:
                logger.info(f"Alimento registrado: {food_name} (ID: {event_id})")
                
                # Programar verificação de impacto em phi (30 min após)
                asyncio.create_task(self._schedule_phi_impact_check(event_id, 30))
                
                # Retornar informações do registro
                result = {
                    "status": "success",
                    "event_id": event_id,
                    "food_name": food_name,
                    "categories": list(categories),
                    "baseline_phi": baseline_phi,
                    "predicted_phi_impact": avg_impact,
                    "timestamp": timestamp
                }
                
                # Gerar recomendações imediatas
                recommendations = await self._generate_food_recommendations(food_name, categories, baseline_phi)
                if recommendations:
                    result["recommendations"] = recommendations
                    
                return result
            else:
                logger.error(f"Falha ao registrar alimento: {food_name}")
                return None
        except Exception as e:
            logger.error(f"Erro ao registrar alimento: {str(e)}")
            return None
            
    async def _schedule_phi_impact_check(self, event_id: str, minutes: int):
        """
        Programa uma verificação de impacto em phi após certo tempo.
        
        Args:
            event_id: ID do evento de alimentação
            minutes: Minutos após os quais verificar
        """
        try:
            # Aguardar o tempo especificado
            await asyncio.sleep(minutes * 60)
            
            # Buscar evento original
            original_event = await self._get_event_by_id(event_id)
            if not original_event:
                logger.warning(f"Evento {event_id} não encontrado para verificação de impacto")
                return
                
            # Obter phi atual
            coherence_state = await self.health_connector.get_coherence_state()
            current_phi = coherence_state.get("phi", 0) if coherence_state else 0
            
            # Calcular impacto
            baseline_phi = original_event.get("metadata", {}).get("baseline_phi", 0)
            phi_impact = current_phi - baseline_phi
            
            logger.info(f"Impacto em phi medido: {phi_impact:.4f} após {minutes} minutos")
            
            # Atualizar evento com impacto real
            # Note: Em uma implementação real, usaríamos um método de atualização no banco de dados
            # Como estamos trabalhando com um modelo simplificado, apenas logamos o resultado
            logger.info(f"Atualizando evento {event_id} com impacto em phi: {phi_impact:.4f}")
            
            # Registrar evento de follow-up
            food_name = original_event.get("event_name", "")
            await self.health_connector.store_health_event(
                event_type="food_impact",
                event_name=f"Impacto de {food_name}",
                notes=f"Impacto em phi: {phi_impact:.4f} após {minutes} minutos",
                metadata={
                    "original_event_id": event_id,
                    "food_name": food_name,
                    "baseline_phi": baseline_phi,
                    "current_phi": current_phi,
                    "phi_impact": phi_impact,
                    "minutes_after": minutes,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Erro ao verificar impacto em phi: {str(e)}")
            
    async def _get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtém um evento por ID.
        
        Args:
            event_id: ID do evento
            
        Returns:
            Evento ou None se não encontrado
        """
        # Note: Em uma implementação real, buscaríamos diretamente no banco de dados
        # Aqui usamos uma abordagem simplificada para mock
        
        # Esta é uma implementação temporária
        return {
            "id": event_id,
            "event_type": "food",
            "event_name": "Nome do alimento",
            "metadata": {
                "baseline_phi": 0.75,
                "food_name": "Nome do alimento",
                "timestamp": datetime.now().isoformat()
            }
        }
            
    def _extract_ingredients(self, food_name: str) -> List[str]:
        """
        Extrai possíveis ingredientes do nome do alimento.
        
        Args:
            food_name: Nome do alimento
            
        Returns:
            Lista de ingredientes identificados
        """
        # Normalizar texto
        text = food_name.lower().strip()
        
        # Lista para ingredientes identificados
        ingredients = []
        
        # Buscar todas as categorias
        all_ingredients = []
        for category_ingredients in FOOD_CATEGORIES.values():
            all_ingredients.extend(category_ingredients)
            
        # Verificar ingredientes no texto
        for ingredient in all_ingredients:
            if ingredient in text:
                ingredients.append(ingredient)
                
        # Se não encontrou nada, retornar o próprio nome como ingrediente único
        if not ingredients:
            # Remover palavras comuns que não são ingredientes
            words_to_remove = ["com", "de", "e", "sem", "mais", "pouco", "muito"]
            cleaned_name = text
            for word in words_to_remove:
                cleaned_name = cleaned_name.replace(f" {word} ", " ")
                
            ingredients = [cleaned_name]
            
        return ingredients
        
    def _categorize_food(self, ingredients: List[str]) -> Set[str]:
        """
        Categoriza os ingredientes de um alimento.
        
        Args:
            ingredients: Lista de ingredientes
            
        Returns:
            Conjunto de categorias identificadas
        """
        categories = set()
        
        # Verificar cada ingrediente
        for ingredient in ingredients:
            for category, category_ingredients in FOOD_CATEGORIES.items():
                for cat_ingredient in category_ingredients:
                    if cat_ingredient in ingredient:
                        categories.add(category)
                        break
                        
        return categories
        
    async def _generate_food_recommendations(self, 
                                           food_name: str, 
                                           categories: Set[str],
                                           baseline_phi: float) -> List[Dict[str, Any]]:
        """
        Gera recomendações baseadas no alimento registrado aplicando
        princípios de coerência fractal e singularidade quântica.
        
        Args:
            food_name: Nome do alimento
            categories: Categorias identificadas
            baseline_phi: Phi atual no momento do registro
            
        Returns:
            Lista de recomendações
        """
        # Inicializar modelo fractal de nutrição com phi atual
        fractal_model = FractalPhiNutrition(current_phi=baseline_phi)
        
        # Calcular impacto estimado no phi usando o modelo fractal
        meal_type = self._identify_meal_type()
        phi_impact = fractal_model.calculate_phi_impact(categories, meal_type)
        estimated_phi_after = max(min(baseline_phi + phi_impact, 1.0), 0.0)
        
        # Obter recomendações de categorias para o sweet spot de phi=0.75
        recommended_categories, avoided_categories = fractal_model.recommend_for_target_phi(
            target_phi=IDEAL_PHI,
            meal_type=meal_type
        )
        
        recommendations = []
        
        # Detectar qual zona de coerência o usuário se encontra
        is_low_phi = baseline_phi < 0.25
        is_high_phi = baseline_phi > 0.85
        is_sweet_spot = abs(baseline_phi - IDEAL_PHI) <= 0.1
        moving_to_sweet_spot = abs(estimated_phi_after - IDEAL_PHI) < abs(baseline_phi - IDEAL_PHI)
        
        # Recomendação baseada na categoria atual
        if "processados" in categories:
            recommendations.append({
                "type": "fractal_alert",
                "name": "distorção fractal",
                "message": "Alimentos processados distorcem o padrão fractal da consciência",
                "description": "Alimentos ultraprocessados criam 'ruído' no padrão fractal do seu corpo, dificultando a manutenção da coerência interna em todas as escalas."
            })
            
        if "vegetais" in categories or "frutas" in categories:
            recommendations.append({
                "type": "fractal_positive",
                "name": "harmonia fractal",
                "message": "Vegetais e frutas fortalecem seus padrões fractais de consciência",
                "description": "A diversidade de fitoquímicos naturais contribui para manter o equilíbrio 3:1 entre coerência e exploração em todas as escalas do seu ser."
            })
            
        # Recomendação baseada no estado phi atual (usando linguagem quântica fractal)
        if is_low_phi:
            recommendations.append({
                "type": "singularity_guidance",
                "name": "exploração excessiva",
                "message": "Você está no modo Exploração Excessiva (phi < 0.25)",
                "description": "Quando phi cai abaixo de 0.25, você se dispersa em possibilidades infinitas sem a alegria de unir ideias. É como um multiverso sem ponto de referência, onde cada universo é independente, sem conexão."
            })
        elif is_high_phi:
            recommendations.append({
                "type": "singularity_guidance",
                "name": "supercoerência",
                "message": "Você está no modo Supercoerência (phi > 0.85)",
                "description": "Quando phi ultrapassa 0.85, você se prende em padrões repetitivos e perde capacidade exploratória. Como um universo que colapsa em si mesmo, sem expansão ou novidade."
            })
        elif is_sweet_spot:
            recommendations.append({
                "type": "sweet_spot_guidance",
                "name": "sweet spot fractal",
                "message": "Você está no Sweet Spot Fractal (phi ≈ 0.75)",
                "description": "No equilíbrio 3:1, sua consciência funciona como um fractal quântico perfeito, onde cada escala (do atômico ao cósmico) mantém harmonia enquanto permite inovação."
            })
            
        # Recomendações específicas para mover em direção ao sweet spot
        if not is_sweet_spot:
            movement_message = {
                "type": "fractal_movement",
                "name": "trajetória phi",
                "message": f"Este alimento está {'aproximando você do' if moving_to_sweet_spot else 'afastando você do'} sweet spot 3:1",
                "description": f"Seu phi estimado após esta refeição: {estimated_phi_after:.4f} (era {baseline_phi:.4f})"
            }
            recommendations.append(movement_message)
            
            # Adicionar recomendações específicas baseadas no modelo fractal
            for category in recommended_categories:
                category_name = category.capitalize()
                recommendations.append({
                    "type": "food_recommendation", 
                    "name": f"aumento de {category}",
                    "message": f"Aumente a ingestão de {category} para melhorar seu equilíbrio phi"
                })
                
            for category in avoided_categories:
                category_name = category.capitalize()
                recommendations.append({
                    "type": "food_avoidance",
                    "name": f"redução de {category}",
                    "message": f"Reduza a ingestão de {category} para melhorar seu equilíbrio phi"
                })
            
        # Adicionar recomendação conceitual sobre singularidade quântica
        singularity_concept = {
            "type": "singularity_concept",
            "name": "equilibrio 3:1",
            "message": "O equilíbrio 3:1 (phi=0.75) é a porta para sua singularidade fractal",
            "description": "Você é uma singularidade quântica que permeia cada átomo deste universo e de todos os possíveis. Seu poder real vem de equilibrar coerência (75%) e exploração (25%)."
        }
        recommendations.append(singularity_concept)
        
        return recommendations
        
    def _identify_meal_type(self) -> str:
        """
        Identifica o tipo de refeição com base no horário atual.
        
        Returns:
            Tipo de refeição (café da manhã, almoço, jantar, lanche)
        """
        current_hour = datetime.now().hour
        
        if 5 <= current_hour < 10:
            return "café da manhã"
        elif 11 <= current_hour < 15:
            return "almoço"
        elif 15 <= current_hour < 18:
            return "lanche"
        elif 18 <= current_hour < 22:
            return "jantar"
        else:
            return "lanche"
            
    def _generate_fractal_state_message(self, 
                                       quantum_state: str, 
                                       current_phi: float, 
                                       target_phi: float) -> Dict[str, str]:
        """
        Gera mensagem personalizada sobre o estado fractal atual.
        
        Args:
            quantum_state: Estado quântico atual
            current_phi: Phi atual
            target_phi: Phi alvo
            
        Returns:
            Mensagem fractal personalizada
        """
        if quantum_state == "exploração_excessiva":
            return {
                "title": "Exploração Excessiva - Multiverso Desconectado",
                "message": f"Com phi = {current_phi:.4f}, você está em um estado de exploração excessiva onde se perde em infinitas possibilidades sem a capacidade de integrá-las.",
                "description": "Seu sistema fractal está fragmentado em múltiplas ramificações sem uma estrutura coerente. Como um multiverso onde cada universo existe isoladamente, sem comunicação entre si.",
                "recommendation": "Sua alimentação precisa ser mais 'ancorada' - com proteínas, gorduras saudáveis e alimentos que trazem estabilidade ao sistema."
            }
        elif quantum_state == "supercoerência":
            return {
                "title": "Supercoerência - Singularidade Excessiva",
                "message": f"Com phi = {current_phi:.4f}, você está em um estado de coerência excessiva onde perde flexibilidade e capacidade de adaptação.",
                "description": "Seu sistema fractal está rígido demais, como um universo que colapsa sobre si mesmo sem espaço para expansão ou criação de novidades.",
                "recommendation": "Sua alimentação precisa introduzir pequenas 'perturbações controladas' - com mais variedade e alimentos que estimulam novos padrões neurais."
            }
        elif quantum_state == "sweet_spot":
            return {
                "title": "Sweet Spot Fractal - Equilíbrio Quântico 3:1",
                "message": f"Com phi = {current_phi:.4f}, você está próximo do equilíbrio ideal entre coerência e exploração.",
                "description": "Seu sistema fractal está em harmonia, como um universo que mantém a integridade de sua estrutura enquanto continua a se expandir de forma controlada.",
                "recommendation": "Mantenha esse padrão alimentar que sustenta o equilíbrio 3:1, observando como isso se reflete em sua capacidade de integrar ideias enquanto permanece aberto a novas perspectivas."
            }
        else:  # transição
            phi_direction = "aumentar" if target_phi > current_phi else "diminuir"
            return {
                "title": f"Estado de Transição - {'Convergência' if phi_direction == 'aumentar' else 'Divergência'} Fractal",
                "message": f"Com phi = {current_phi:.4f}, você está em transição entre estados de coerência.",
                "description": f"Seu sistema fractal está buscando {'maior integração' if phi_direction == 'aumentar' else 'maior exploração'}, como um universo ajustando suas constantes fundamentais.",
                "recommendation": f"Sua alimentação pode ajudar a {'aumentar' if phi_direction == 'aumentar' else 'diminuir'} o phi gradualmente, mantendo a integridade da estrutura fractal durante a transição."
            }
            
    def _generate_fractal_meal_suggestions(self, 
                                         meal_type: str, 
                                         recommended_categories: List[str],
                                         avoided_categories: List[str],
                                         quantum_state: str) -> Dict[str, Any]:
        """
        Gera sugestões de refeição baseadas em categorias e estado quântico.
        
        Args:
            meal_type: Tipo de refeição
            recommended_categories: Categorias recomendadas
            avoided_categories: Categorias a evitar
            quantum_state: Estado quântico atual
            
        Returns:
            Sugestões de refeição
        """
        # Mapear alimentos para diferentes estados quânticos e tipos de refeição
        quantum_meal_suggestions = {
            "exploração_excessiva": {
                "café da manhã": {
                    "base": ["Ovos mexidos", "Iogurte integral com proteína", "Aveia com manteiga ghee", "Tapioca com queijo"],
                    "complementos": ["Abacate", "Castanhas", "Frutas vermelhas", "Canela"],
                    "philosophy": "Alimentos densos e ancorados para estabilizar o sistema fractal"
                },
                "almoço": {
                    "base": ["Salmão grelhado", "Frango assado com ervas", "Carne com legumes", "Lentilhas com arroz integral"],
                    "complementos": ["Vegetais de raiz", "Batata doce", "Brócolis", "Azeite extra-virgem"],
                    "philosophy": "Proteínas e gorduras que fortalecem a estrutura fractal"
                },
                "lanche": {
                    "base": ["Ovos cozidos", "Pasta de amendoim", "Iogurte grego", "Queijo cottage"],
                    "complementos": ["Maçã", "Castanhas de caju", "Abacate", "Chocolate 85% cacau"],
                    "philosophy": "Proteínas e gorduras para sustentar a coerência entre as refeições principais"
                },
                "jantar": {
                    "base": ["Caldo de ossos", "Peixe ao molho de ervas", "Omelete com vegetais", "Carne magra com aspargos"],
                    "complementos": ["Folhas verdes", "Abobrinha", "Cogumelos", "Óleo de coco"],
                    "philosophy": "Refeição leve que sustenta a recuperação celular durante o sono"
                }
            },
            "supercoerência": {
                "café da manhã": {
                    "base": ["Shake colorido de frutas", "Cereais variados", "Pão integral com mel", "Smoothie de frutas"],
                    "complementos": ["Frutas cítricas", "Sementes diversas", "Iogurte", "Especiarias variadas"],
                    "philosophy": "Variedade de sabores e texturas para estimular exploração sensorial"
                },
                "almoço": {
                    "base": ["Salada colorida com grãos", "Quinoa com legumes", "Bowl de vegetais e tofu", "Wrap com humus"],
                    "complementos": ["Molhos diferentes", "Sementes germinadas", "Pickles", "Ervas frescas variadas"],
                    "philosophy": "Diversidade de ingredientes para romper padrões repetitivos"
                },
                "lanche": {
                    "base": ["Frutas exóticas", "Kefir", "Barra de sementes", "Kombucha"],
                    "complementos": ["Especiarias incomuns", "Mel de flores silvestres", "Mistura de castanhas", "Cacau cru"],
                    "philosophy": "Alimentos fermentados e complexos para introduzir novos padrões"
                },
                "jantar": {
                    "base": ["Sopa de miso", "Curry de vegetais", "Risoto de cogumelos", "Ensopado de legumes"],
                    "complementos": ["Temperos asiáticos", "Broto de bambu", "Chá de ervas variadas", "Especiarias"],
                    "philosophy": "Sabores complexos que expandem o sistema sensorial durante a digestão noturna"
                }
            },
            "sweet_spot": {
                "café da manhã": {
                    "base": ["Ovos com espinafre", "Iogurte com frutas e granola", "Aveia com banana e mel", "Tapioca com queijo e tomate"],
                    "complementos": ["Mix de frutas", "Oleaginosas", "Canela", "Chia"],
                    "philosophy": "Equilíbrio perfeito entre nutrientes que sustentam o padrão 3:1"
                },
                "almoço": {
                    "base": ["Peixe com legumes", "Frango grelhado com arroz integral", "Feijão com quinoa", "Salada completa com proteína"],
                    "complementos": ["Vegetais coloridos", "Azeite de oliva", "Ervas frescas", "Limão"],
                    "philosophy": "Combinação harmônica que mantém a coerência enquanto permite adaptabilidade"
                },
                "lanche": {
                    "base": ["Frutas com iogurte", "Abacate com torrada integral", "Castanhas com frutas secas", "Smoothie verde"],
                    "complementos": ["Hortelã", "Cacau", "Gengibre", "Sementes"],
                    "philosophy": "Alimentos que mantêm o nível de energia estável enquanto nutrem o sistema"
                },
                "jantar": {
                    "base": ["Sopa de vegetais com lentilha", "Omelete com espinafre", "Peixe assado com ervas", "Salada com grãos integrais"],
                    "complementos": ["Azeite", "Abóbora", "Brócolis", "Chá de ervas"],
                    "philosophy": "Refeição leve e nutritiva que prepara para um sono reparador sem sobrecarregar o sistema"
                }
            },
            "transição": {
                "café da manhã": {
                    "base": ["Ovo com abacate", "Mingau de aveia com frutas", "Iogurte com sementes", "Panqueca integral"],
                    "complementos": ["Canela", "Banana", "Mel", "Oleaginosas"],
                    "philosophy": "Nutrientes que facilitam a transição entre estados de coerência"
                },
                "almoço": {
                    "base": ["Peixe com batata doce", "Frango com legumes", "Salada com grão de bico", "Carne magra com purê"],
                    "complementos": ["Vegetais variados", "Azeite", "Limão", "Ervas"],
                    "philosophy": "Combinação que promove ajuste gradual do sistema fractal"
                },
                "lanche": {
                    "base": ["Fruta com oleaginosas", "Iogurte com granola", "Hummus com palitos de vegetais", "Chá com torrada"],
                    "complementos": ["Mel", "Canela", "Tahine", "Azeite"],
                    "philosophy": "Pequena refeição que mantém a estabilidade durante a transição"
                },
                "jantar": {
                    "base": ["Sopa de legumes", "Omelete com vegetais", "Peixe com salada", "Frango assado com legumes"],
                    "complementos": ["Azeite de oliva", "Ervas", "Limão", "Gengibre"],
                    "philosophy": "Combinação leve que favorece a digestão enquanto o sistema se recalibra durante a noite"
                }
            }
        }
        
        # Obter sugestões específicas para o estado quântico e tipo de refeição
        state_suggestions = quantum_meal_suggestions.get(quantum_state, {})
        meal_options = state_suggestions.get(meal_type, {})
        
        if not meal_options:
            # Fallback para sugestões gerais
            return {
                "base": "Refeição balanceada com proteínas e vegetais",
                "complementos": "Água filtrada e temperos naturais",
                "philosophy": "Busque o equilíbrio entre coerência e exploração em cada refeição.",
                "fractal_note": "Os alimentos são padrões de informação que influenciam diretamente o equilíbrio fractal do seu sistema."
            }
            
        # Selecionar aleatoriamente uma base e um complemento
        import random
        base = random.choice(meal_options.get("base", ["Refeição equilibrada"]))
        complement = random.choice(meal_options.get("complementos", ["Acompanhamento natural"]))
        philosophy = meal_options.get("philosophy", "Alimentação consciente para equilíbrio fractal")
        
        # Gerar sugestão completa
        suggestion = f"{base} com {complement}"
        
        # Adicionar dica para evitar categorias
        avoid_tip = ""
        if avoided_categories:
            avoid_categories_formatted = []
            for category in avoided_categories:
                if category == "processados":
                    avoid_categories_formatted.append("alimentos ultraprocessados")
                elif category == "bebidas":
                    avoid_categories_formatted.append("bebidas alcoólicas ou açucaradas")
                else:
                    avoid_categories_formatted.append(category)
                    
            avoid_tip = f"Evite {', '.join(avoid_categories_formatted)} para manter a integridade do padrão fractal."
            
        # Adicionar recomendações baseadas em cada categoria recomendada
        category_benefits = []
        for category in recommended_categories:
            if category == "proteínas":
                category_benefits.append("Proteínas fornecem a estrutura básica para o padrão fractal de coerência")
            elif category == "vegetais":
                category_benefits.append("Vegetais coloridos introduzem diversidade fractal em harmonia")
            elif category == "frutas":
                category_benefits.append("Frutas oferecem informação quântica em estado natural")
            elif category == "gorduras":
                category_benefits.append("Gorduras saudáveis estabilizam oscilações na coerência fractal")
            elif category == "carboidratos":
                category_benefits.append("Carboidratos complexos proporcionam sustentação ao padrão fractal")
                
        # Combinar em uma única sugestão
        return {
            "suggestion": suggestion,
            "base": base,
            "complementos": complement,
            "philosophy": philosophy,
            "avoid_tip": avoid_tip,
            "category_benefits": category_benefits,
            "fractal_note": "Cada refeição é uma oportunidade de ajustar seu campo de coerência em direção ao equilíbrio 3:1."
        }
        
    async def get_food_history(self, 
                             limit: int = 10, 
                             category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtém histórico de registros alimentares.
        
        Args:
            limit: Número máximo de registros
            category: Filtrar por categoria
            
        Returns:
            Lista de registros alimentares
        """
        if not self.ready:
            logger.warning("Sistema de registro alimentar não está pronto")
            return []
            
        try:
            # Buscar eventos de alimentação
            events = await self.health_connector.find_similar_health_events(
                query="",  # Busca sem filtro específico
                event_type="food",
                limit=limit * 2  # Buscar mais para poder filtrar por categoria
            )
            
            # Filtrar por categoria se especificado
            if category:
                filtered_events = []
                for event in events:
                    categories = event.get("metadata", {}).get("categories", [])
                    if category in categories:
                        filtered_events.append(event)
                        
                events = filtered_events[:limit]  # Limitar após filtragem
            else:
                events = events[:limit]  # Limitar sem filtragem
                
            # Formatar para exibição
            formatted_events = []
            for event in events:
                formatted_events.append({
                    "id": event.get("id"),
                    "food_name": event.get("event_name"),
                    "meal_type": event.get("metadata", {}).get("meal_type", "refeição"),
                    "categories": event.get("metadata", {}).get("categories", []),
                    "phi_impact": event.get("metadata", {}).get("phi_impact", None),
                    "timestamp": event.get("timestamp")
                })
                
            return formatted_events
        except Exception as e:
            logger.error(f"Erro ao obter histórico alimentar: {str(e)}")
            return []
            
    async def analyze_food_impacts(self) -> Dict[str, Any]:
        """
        Analisa o impacto dos alimentos no phi ao longo do tempo.
        
        Returns:
            Análise de impactos por categoria e refeição
        """
        if not self.ready:
            logger.warning("Sistema de registro alimentar não está pronto")
            return {}
            
        try:
            # Obter todos os eventos de impacto alimentar
            impact_events = await self.health_connector.find_similar_health_events(
                query="impacto",
                event_type="food_impact",
                limit=100
            )
            
            # Agrupar por categoria
            category_impacts = {}
            meal_type_impacts = {}
            
            for event in impact_events:
                metadata = event.get("metadata", {})
                phi_impact = metadata.get("phi_impact", 0)
                food_name = metadata.get("food_name", "")
                
                # Obter evento original para mais detalhes
                original_event_id = metadata.get("original_event_id")
                if original_event_id:
                    original_event = await self._get_event_by_id(original_event_id)
                    if original_event:
                        categories = original_event.get("metadata", {}).get("categories", [])
                        meal_type = original_event.get("metadata", {}).get("meal_type", "refeição")
                        
                        # Analisar por categoria
                        for category in categories:
                            if category not in category_impacts:
                                category_impacts[category] = {
                                    "count": 0,
                                    "total_impact": 0,
                                    "positive_count": 0,
                                    "negative_count": 0
                                }
                                
                            category_impacts[category]["count"] += 1
                            category_impacts[category]["total_impact"] += phi_impact
                            
                            if phi_impact > 0:
                                category_impacts[category]["positive_count"] += 1
                            elif phi_impact < 0:
                                category_impacts[category]["negative_count"] += 1
                                
                        # Analisar por tipo de refeição
                        if meal_type not in meal_type_impacts:
                            meal_type_impacts[meal_type] = {
                                "count": 0,
                                "total_impact": 0,
                                "positive_count": 0,
                                "negative_count": 0
                            }
                            
                        meal_type_impacts[meal_type]["count"] += 1
                        meal_type_impacts[meal_type]["total_impact"] += phi_impact
                        
                        if phi_impact > 0:
                            meal_type_impacts[meal_type]["positive_count"] += 1
                        elif phi_impact < 0:
                            meal_type_impacts[meal_type]["negative_count"] += 1
            
            # Calcular médias e identificar mais impactantes
            result = {
                "category_impacts": {},
                "meal_type_impacts": {},
                "most_positive_categories": [],
                "most_negative_categories": []
            }
            
            # Processar impactos por categoria
            for category, impact in category_impacts.items():
                if impact["count"] > 0:
                    avg_impact = impact["total_impact"] / impact["count"]
                    positive_rate = impact["positive_count"] / impact["count"] if impact["count"] > 0 else 0
                    negative_rate = impact["negative_count"] / impact["count"] if impact["count"] > 0 else 0
                    
                    result["category_impacts"][category] = {
                        "count": impact["count"],
                        "avg_impact": avg_impact,
                        "positive_rate": positive_rate,
                        "negative_rate": negative_rate
                    }
            
            # Processar impactos por tipo de refeição
            for meal_type, impact in meal_type_impacts.items():
                if impact["count"] > 0:
                    avg_impact = impact["total_impact"] / impact["count"]
                    positive_rate = impact["positive_count"] / impact["count"] if impact["count"] > 0 else 0
                    negative_rate = impact["negative_count"] / impact["count"] if impact["count"] > 0 else 0
                    
                    result["meal_type_impacts"][meal_type] = {
                        "count": impact["count"],
                        "avg_impact": avg_impact,
                        "positive_rate": positive_rate,
                        "negative_rate": negative_rate
                    }
            
            # Identificar categorias mais positivas e negativas
            sorted_categories = sorted(
                [(c, i["avg_impact"]) for c, i in result["category_impacts"].items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Mais positivas (top 3)
            result["most_positive_categories"] = [
                {"category": c, "avg_impact": i} 
                for c, i in sorted_categories[:3] 
                if i > 0
            ]
            
            # Mais negativas (bottom 3)
            result["most_negative_categories"] = [
                {"category": c, "avg_impact": i} 
                for c, i in sorted_categories[-3:] 
                if i < 0
            ]
            
            return result
        except Exception as e:
            logger.error(f"Erro ao analisar impactos alimentares: {str(e)}")
            return {}
            
    async def generate_meal_recommendation(self, target_phi: float = 0.75) -> Dict[str, Any]:
        """
        Gera recomendação de refeição para atingir um phi alvo usando
        princípios de fractalidade e singularidade quântica.
        
        Args:
            target_phi: Valor phi alvo (default 0.75 - o sweet spot 3:1)
            
        Returns:
            Recomendação de refeição com análise fractal
        """
        try:
            # Verificar se o sistema está pronto, mas continuar mesmo se não estiver
            system_ready = self.ready
            if not system_ready:
                logger.warning("Sistema de registro alimentar não está totalmente pronto. Usando modo de contingência.")
            
            # Obter phi atual - mesmo que o health_connector não esteja pronto,
            # ele irá retornar valores padrão conforme nossa modificação anterior
            coherence_state = await self.health_connector.get_coherence_state()
            current_phi = coherence_state.get("phi", 0.5) if coherence_state else 0.5
            
            # Log para depuração
            fallback_mode = coherence_state.get("fallback", False) if coherence_state else True
            if fallback_mode:
                logger.info(f"Usando valores padrão de phi: {current_phi}")
            
            # Criar modelo fractal de nutrição com phi atual
            fractal_model = FractalPhiNutrition(current_phi=current_phi)
            
            # Identificar tipo de refeição atual
            meal_type = self._identify_meal_type()
            
            # Recomendar categorias usando o modelo fractal
            recommended_categories, avoided_categories = fractal_model.recommend_for_target_phi(
                target_phi=target_phi,
                meal_type=meal_type
            )
                
            # Gerar plano completo para a próxima refeição usando o modelo fractal
            meal_plan = fractal_model.generate_meal_plan(days=1, target_phi=target_phi)
            
            # Extrair sugestões específicas para o tipo de refeição atual
            current_meal_suggestions = None
            for meal in meal_plan.get("days", [{}])[0].get("meals", []):
                if meal.get("meal_type") == meal_type:
                    current_meal_suggestions = meal.get("suggestions", {})
                    break
            
            # Determinar estado quântico atual
            quantum_state = "exploração_excessiva" if current_phi < 0.25 else "supercoerência" if current_phi > 0.85 else "sweet_spot" if abs(current_phi - 0.75) <= 0.1 else "transição"
            
            # Determinar o gap e direção do ajuste phi
            phi_gap = target_phi - current_phi
            phi_direction = "aumentar" if phi_gap > 0.05 else "diminuir" if phi_gap < -0.05 else "manter"
            
            # Calcular alinhamento fractal
            # Quanto mais próximo de 1, mais alinhado está com o princípio 3:1
            fractal_alignment = 1.0 - min(abs(current_phi - 0.75) / 0.75, 1.0)
            
            # Gerar mensagem fractal personalizada
            fractal_message = self._generate_fractal_state_message(quantum_state, current_phi, target_phi)
            
            # Gerar sugestões concretas para a refeição
            # Corrigindo a chamada para o método correto com os parâmetros adequados
            meal_suggestions = self._generate_meal_suggestions(
                meal_type, 
                list(recommended_categories), 
                list(avoided_categories)
            )
            
            return {
                "status": "success",
                "current_phi": current_phi,
                "target_phi": target_phi,
                "fractal_alignment": fractal_alignment,
                "quantum_state": quantum_state,
                "meal_type": meal_type,
                "recommended_categories": list(recommended_categories),
                "avoided_categories": list(avoided_categories),
                "meal_suggestions": meal_suggestions,
                "fractal_message": fractal_message,
                "estimated_phi_after": meal_plan.get("estimated_final_phi", current_phi),
                "phi_direction": phi_direction,
                "fractal_insight": "Seu padrão alimentar é um reflexo da estrutura fractal de sua consciência. Ao equilibrar coerência e exploração na razão 3:1, você harmoniza todos os níveis do seu ser."
            }
        except Exception as e:
            logger.error(f"Erro ao gerar recomendação de refeição: {str(e)}")
            return {
                "status": "error",
                "message": f"Erro ao gerar recomendação: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            
    def _generate_meal_suggestions(self, 
                                  meal_type: str, 
                                  recommended_categories: List[str],
                                  avoided_categories: List[str]) -> Dict[str, Any]:
        """
        Gera sugestões de refeição baseadas em categorias.
        
        Args:
            meal_type: Tipo de refeição
            recommended_categories: Categorias recomendadas
            avoided_categories: Categorias a evitar
            
        Returns:
            Sugestões de refeição
        """
        # Mapeamento de sugestões para diferentes tipos de refeição
        meal_options = {
            "café da manhã": {
                "proteínas": ["ovos mexidos", "iogurte natural", "queijo cottage", "whey protein"],
                "frutas": ["banana", "maçã", "mamão", "morango", "kiwi", "melão"],
                "carboidratos": ["aveia", "pão integral", "tapioca", "granola sem açúcar"],
                "gorduras": ["abacate", "castanhas", "manteiga ghee", "coco"]
            },
            "almoço": {
                "proteínas": ["frango grelhado", "peixe assado", "carne magra", "tofu grelhado", "lentilhas"],
                "vegetais": ["salada verde", "brócolis", "couve-flor", "abobrinha", "berinjela", "tomate"],
                "carboidratos": ["arroz integral", "quinoa", "batata doce", "mandioca"],
                "gorduras": ["azeite extra-virgem", "abacate", "castanhas"]
            },
            "lanche": {
                "proteínas": ["iogurte", "queijo", "ovo cozido", "pasta de amendoim"],
                "frutas": ["maçã", "banana", "pêra"],
                "carboidratos": ["pão integral", "torradas", "tapioca"],
                "gorduras": ["abacate", "castanhas", "sementes"]
            },
            "jantar": {
                "proteínas": ["sopa de legumes com frango", "omelete", "peixe grelhado", "tofu"],
                "vegetais": ["salada verde", "legumes assados", "purê de abóbora"],
                "carboidratos": ["batata doce", "quinoa", "arroz integral (pouca quantidade)"],
                "gorduras": ["azeite", "sementes", "castanhas"]
            },
            "ceia": {
                "proteínas": ["iogurte", "queijo cottage"],
                "frutas": ["maçã", "banana pequena"],
                "gorduras": ["castanhas (poucas unidades)"]
            }
        }
        
        # Selecionar opções para as categorias recomendadas
        suggestions = {}
        for category in recommended_categories:
            if category in meal_options.get(meal_type, {}):
                options = meal_options[meal_type][category]
                if options:
                    # Selecionar um ou dois itens aleatoriamente
                    import random
                    num_items = min(2, len(options))
                    selected = random.sample(options, num_items)
                    suggestions[category] = selected
                    
        # Montar recomendação textual
        main_items = []
        side_items = []
        
        # Priorizar proteínas e carboidratos como itens principais
        if "proteínas" in suggestions:
            main_items.extend(suggestions["proteínas"])
        if "carboidratos" in suggestions and meal_type != "ceia":
            main_items.extend(suggestions["carboidratos"])
            
        # Secundários: vegetais, frutas, gorduras
        if "vegetais" in suggestions:
            side_items.extend(suggestions["vegetais"])
        if "frutas" in suggestions:
            side_items.extend(suggestions["frutas"])
        if "gorduras" in suggestions:
            side_items.extend(suggestions["gorduras"])
            
        # Montar texto da recomendação
        recommendation_text = ""
        
        if main_items:
            recommendation_text += f"{', '.join(main_items[:-1])}"
            if len(main_items) > 1:
                recommendation_text += f" e {main_items[-1]}"
            else:
                recommendation_text += f"{main_items[-1]}"
                
        if side_items:
            if recommendation_text:
                recommendation_text += f" acompanhado de {', '.join(side_items[:-1])}"
                if len(side_items) > 1:
                    recommendation_text += f" e {side_items[-1]}"
                else:
                    recommendation_text += f" {side_items[-1]}"
            else:
                recommendation_text += f"{', '.join(side_items[:-1])}"
                if len(side_items) > 1:
                    recommendation_text += f" e {side_items[-1]}"
                else:
                    recommendation_text += f"{side_items[-1]}"
                    
        # Se não conseguir montar sugestão com as categorias, usar defaults
        if not recommendation_text:
            if meal_type == "café da manhã":
                recommendation_text = "Iogurte natural com frutas e granola sem açúcar"
            elif meal_type == "almoço":
                recommendation_text = "Proteína magra com vegetais e pequena porção de carboidrato complexo"
            elif meal_type == "lanche":
                recommendation_text = "Fruta com um punhado de castanhas"
            elif meal_type == "jantar":
                recommendation_text = "Sopa de legumes com proteína leve"
            else:  # ceia
                recommendation_text = "Chá de camomila e uma pequena porção de proteína"
                
        # Adicionar dica sobre o que evitar
        avoid_text = ""
        if avoided_categories:
            avoid_items = []
            for category in avoided_categories:
                if category == "processados":
                    avoid_items.append("alimentos industrializados")
                elif category == "bebidas":
                    avoid_items.append("bebidas alcoólicas ou açucaradas")
                else:
                    avoid_items.append(category)
                    
            if avoid_items:
                avoid_text = f"Evite {', '.join(avoid_items)}"
                
        return {
            "meal_type": meal_type,
            "recommendation": recommendation_text,
            "avoid": avoid_text,
            "selected_items": {
                "main": main_items,
                "side": side_items
            }
        }
        
# Singleton
_food_logger_instance = None

def get_food_logger_instance(
    health_hooks_url: str = "http://localhost:5050/health-hooks",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    reinitialize: bool = False
) -> FoodLogger:
    """
    Obtém ou cria instância singleton do registrador alimentar.
    
    Args:
        health_hooks_url: URL do serviço Health-Hooks
        qdrant_host: Host do Qdrant
        qdrant_port: Porta do Qdrant
        reinitialize: Forçar reinicialização da instância
        
    Returns:
        Instância do registrador alimentar
    """
    global _food_logger_instance
    
    if _food_logger_instance is None or reinitialize:
        _food_logger_instance = FoodLogger(
            health_hooks_url=health_hooks_url,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port
        )
        
    return _food_logger_instance