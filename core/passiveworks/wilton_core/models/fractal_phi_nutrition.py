#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo Fractal de Nutrição para WiltonOS

Este módulo implementa um modelo matemático fractal para analisar o impacto
dos alimentos no nível de coerência (phi) do usuário. Baseado no conceito
de que o equilíbrio 3:1 (phi = 0.75) representa o "sweet spot" para a 
consciência fractal, o modelo avalia como diferentes padrões alimentares
podem aproximar ou afastar o usuário desse equilíbrio ideal.

O modelo trata cada alimento como um padrão fractal que se replica em
diferentes escalas no corpo, criando ou perturbando a harmonia global
do sistema.
"""

import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta

# Sweet spot de phi para equilíbrio fractal
IDEAL_PHI = 0.75
# Faixa de tolerância para phi ideal
PHI_TOLERANCE = 0.1
# Faixa para phi baixo
LOW_PHI_THRESHOLD = 0.25
# Faixa para phi alto
HIGH_PHI_THRESHOLD = 0.85

class FractalPhiNutrition:
    """
    Implementa modelo fractal para análise nutricional e seu impacto em phi.
    
    Este modelo considera:
    1. A relação entre categorias alimentares e seus impactos em phi
    2. Padrões fractais de resposta nutricional ao longo do tempo
    3. A busca pelo "sweet spot" de phi = 0.75
    """
    
    def __init__(self, current_phi: float = 0.5):
        """
        Inicializa o modelo com phi atual.
        
        Args:
            current_phi: Valor phi atual do sistema
        """
        self.current_phi = current_phi
        # Pesos de categorias alimentares no modelo
        self.category_weights = {
            "proteínas": 0.2,
            "carboidratos": 0.1,
            "gorduras": 0.15,
            "vegetais": 0.25,
            "frutas": 0.2,
            "condimentos": 0.05,
            "bebidas": -0.1,  # Impacto negativo
            "processados": -0.3  # Impacto muito negativo
        }
        
    def calculate_phi_impact(self, 
                           categories: Set[str],
                           meal_type: str,
                           phi_history: Optional[List[float]] = None) -> float:
        """
        Calcula o impacto de um conjunto de categorias alimentares em phi.
        
        Args:
            categories: Conjunto de categorias alimentares
            meal_type: Tipo de refeição
            phi_history: Histórico recente de valores phi (opcional)
            
        Returns:
            Impacto estimado em phi (positivo = aumento, negativo = diminuição)
        """
        # Calcular impacto base somando pesos das categorias
        base_impact = sum(self.category_weights.get(category, 0) for category in categories)
        
        # Aplicar modificador baseado no tipo de refeição
        meal_modifier = self._get_meal_type_modifier(meal_type)
        
        # Modificador para aproximar do sweet spot de phi
        direction_modifier = self._calculate_direction_modifier()
        
        # Modificador fractal baseado em histórico
        fractal_modifier = 1.0
        if phi_history:
            fractal_modifier = self._calculate_fractal_modifier(phi_history)
            
        # Calcular impacto final
        impact = base_impact * meal_modifier * direction_modifier * fractal_modifier
        
        # Limitar impacto a um range razoável
        return max(min(impact, 0.3), -0.3)
        
    def _get_meal_type_modifier(self, meal_type: str) -> float:
        """
        Retorna modificador para o tipo de refeição.
        
        Args:
            meal_type: Tipo de refeição
            
        Returns:
            Modificador para impacto
        """
        # Diferentes refeições têm diferentes impactos em phi
        modifiers = {
            "café da manhã": 1.2,  # Maior impacto (início do dia)
            "almoço": 1.0,  # Impacto padrão
            "lanche": 0.7,  # Impacto menor
            "jantar": 0.8,  # Impacto médio
            "ceia": 0.5   # Impacto menor (fim do dia)
        }
        
        return modifiers.get(meal_type.lower(), 1.0)
        
    def _calculate_direction_modifier(self) -> float:
        """
        Calcula modificador para direcionar phi para o sweet spot.
        
        Returns:
            Modificador de direção
        """
        # Se estamos abaixo do sweet spot, favorecer impactos positivos
        if self.current_phi < IDEAL_PHI - PHI_TOLERANCE:
            gap = IDEAL_PHI - self.current_phi
            # Quanto mais longe do ideal, maior o modificador
            return 1.0 + min(gap, 0.5)
            
        # Se estamos acima do sweet spot, favorecer impactos negativos
        elif self.current_phi > IDEAL_PHI + PHI_TOLERANCE:
            gap = self.current_phi - IDEAL_PHI
            # Inverter o sinal para impactos negativos
            return 1.0 - min(gap, 0.5)
            
        # Se estamos no sweet spot, modificador neutro
        else:
            return 1.0
            
    def _calculate_fractal_modifier(self, phi_history: List[float]) -> float:
        """
        Calcula modificador fractal baseado em padrões no histórico phi.
        
        Args:
            phi_history: Histórico de valores phi
            
        Returns:
            Modificador fractal
        """
        if len(phi_history) < 3:
            return 1.0
            
        # Calcular diferenças entre valores consecutivos
        diffs = [phi_history[i+1] - phi_history[i] for i in range(len(phi_history)-1)]
        
        # Verificar se há um padrão fractal (auto-similaridade em diferentes escalas)
        if self._has_fractal_pattern(diffs):
            # Reforçar padrão fractal se estiver se aproximando do sweet spot
            if abs(self.current_phi - IDEAL_PHI) < abs(phi_history[-1] - IDEAL_PHI):
                return 1.3  # Aumentar efeito
            else:
                return 0.7  # Diminuir efeito
                
        return 1.0
        
    def _has_fractal_pattern(self, diffs: List[float]) -> bool:
        """
        Verifica se há um padrão fractal nas diferenças.
        
        Args:
            diffs: Lista de diferenças entre valores consecutivos
            
        Returns:
            True se detectar padrão fractal, False caso contrário
        """
        if len(diffs) < 3:
            return False
            
        # Método simples: verificar se as razões entre diferenças são aproximadamente constantes
        ratios = [abs(diffs[i+1] / diffs[i]) if diffs[i] != 0 else 1.0 for i in range(len(diffs)-1)]
        avg_ratio = sum(ratios) / len(ratios)
        
        # Verificar se as razões estão próximas da média (indicando auto-similaridade)
        for ratio in ratios:
            if abs(ratio - avg_ratio) > 0.3:  # Tolerância
                return False
                
        return True
        
    def categorize_food(self, food_name: str, ingredients: List[str]) -> Set[str]:
        """
        Categoriza alimento com base no nome e ingredientes.
        
        Args:
            food_name: Nome do alimento
            ingredients: Lista de ingredientes
            
        Returns:
            Conjunto de categorias
        """
        # Palavras-chave para cada categoria
        category_keywords = {
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
        
        # Conjunto para armazenar categorias identificadas
        categories = set()
        
        # Normalizar texto
        food_name_lower = food_name.lower()
        ingredients_lower = [ing.lower() for ing in ingredients]
        
        # Verificar palavras-chave no nome do alimento
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in food_name_lower:
                    categories.add(category)
                    break
                    
        # Verificar palavras-chave nos ingredientes
        for ingredient in ingredients_lower:
            for category, keywords in category_keywords.items():
                for keyword in keywords:
                    if keyword in ingredient:
                        categories.add(category)
                        break
                        
        # Se nenhuma categoria foi identificada, adicionar "outro"
        if not categories:
            categories.add("outro")
            
        return categories
        
    def calculate_phi_after_meal(self, 
                                meal_categories: Set[str], 
                                meal_type: str) -> float:
        """
        Calcula phi estimado após uma refeição.
        
        Args:
            meal_categories: Categorias do alimento
            meal_type: Tipo de refeição
            
        Returns:
            Phi estimado após a refeição
        """
        impact = self.calculate_phi_impact(meal_categories, meal_type)
        new_phi = self.current_phi + impact
        
        # Garantir que phi está no intervalo [0, 1]
        return max(min(new_phi, 1.0), 0.0)
        
    def recommend_for_target_phi(self, 
                                target_phi: float = IDEAL_PHI,
                                meal_type: str = "refeição") -> Tuple[Set[str], Set[str]]:
        """
        Recomenda categorias alimentares para atingir phi alvo.
        
        Args:
            target_phi: Phi alvo
            meal_type: Tipo de refeição
            
        Returns:
            Tupla com (categorias recomendadas, categorias a evitar)
        """
        # Calcular gap entre phi atual e alvo
        phi_gap = target_phi - self.current_phi
        
        recommended = set()
        avoided = set()
        
        # Selecionar categorias com base na direção necessária
        if phi_gap > 0.05:  # Precisamos aumentar phi
            # Ordenar categorias por impacto positivo
            positive_categories = [(cat, weight) for cat, weight in self.category_weights.items() if weight > 0]
            positive_categories.sort(key=lambda x: x[1], reverse=True)
            
            # Adicionar top categorias positivas às recomendações
            for cat, _ in positive_categories[:3]:
                recommended.add(cat)
                
            # Evitar categorias negativas
            for cat, weight in self.category_weights.items():
                if weight < 0:
                    avoided.add(cat)
                    
        elif phi_gap < -0.05:  # Precisamos diminuir phi
            # Ordenar categorias por impacto negativo
            negative_categories = [(cat, weight) for cat, weight in self.category_weights.items() if weight < 0]
            negative_categories.sort(key=lambda x: x[1])
            
            # Adicionar categorias negativas às recomendações
            for cat, _ in negative_categories[:2]:
                recommended.add(cat)
                
            # Evitar categorias muito positivas
            positive_categories = [(cat, weight) for cat, weight in self.category_weights.items() if weight > 0.2]
            for cat, _ in positive_categories:
                avoided.add(cat)
                
        else:  # Manter phi atual
            # Recomendar mix equilibrado
            balanced_categories = ["proteínas", "vegetais"]
            for cat in balanced_categories:
                recommended.add(cat)
                
            # Evitar extremos
            avoided.add("processados")
            
        # Ajustar para tipo de refeição
        if meal_type == "café da manhã":
            recommended.add("frutas")
        elif meal_type == "almoço":
            recommended.add("proteínas")
            recommended.add("vegetais")
        elif meal_type == "jantar":
            # Para jantar, preferir mais leve
            if "carboidratos" in recommended:
                recommended.remove("carboidratos")
                
        return recommended, avoided
        
    def calculate_fractal_coherence(self, 
                                   phi_history: List[float], 
                                   meal_history: List[Set[str]]) -> float:
        """
        Calcula índice de coerência fractal entre histórico de phi e alimentação.
        
        Args:
            phi_history: Histórico de valores phi
            meal_history: Histórico de categorias alimentares
            
        Returns:
            Índice de coerência fractal [0-1]
        """
        if len(phi_history) < 3 or len(meal_history) < 3:
            return 0.5  # Valor neutro se não houver dados suficientes
            
        # Verificar se há padrão fractal nos valores de phi
        phi_pattern_strength = self._calculate_fractal_pattern_strength(phi_history)
        
        # Verificar regularidade nas categorias alimentares
        category_regularity = self._calculate_category_regularity(meal_history)
        
        # Verificar proximidade do sweet spot
        sweet_spot_proximity = 1.0 - min(abs(phi_history[-1] - IDEAL_PHI) / IDEAL_PHI, 1.0)
        
        # Combinar métricas
        coherence = (phi_pattern_strength * 0.4 + 
                     category_regularity * 0.3 + 
                     sweet_spot_proximity * 0.3)
                     
        return coherence
        
    def _calculate_fractal_pattern_strength(self, values: List[float]) -> float:
        """
        Calcula força de um padrão fractal em uma série temporal.
        
        Args:
            values: Série temporal de valores
            
        Returns:
            Força do padrão fractal [0-1]
        """
        if len(values) < 3:
            return 0.0
            
        # Calcular diferenças entre valores consecutivos
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        
        # Calcular autocorrelação em diferentes escalas (indicador de auto-similaridade)
        autocorr_scores = []
        for lag in range(1, min(5, len(diffs) // 2)):
            autocorr = self._autocorrelation(diffs, lag)
            autocorr_scores.append(abs(autocorr))
            
        if not autocorr_scores:
            return 0.0
            
        # Calcular média das autocorrelações como indicador de força fractal
        return min(sum(autocorr_scores) / len(autocorr_scores), 1.0)
        
    def _autocorrelation(self, values: List[float], lag: int) -> float:
        """
        Calcula autocorrelação simples com lag específico.
        
        Args:
            values: Série de valores
            lag: Deslocamento para autocorrelação
            
        Returns:
            Valor de autocorrelação [-1 a 1]
        """
        n = len(values)
        if n <= lag:
            return 0.0
            
        # Calcular média
        mean = sum(values) / n
        
        # Calcular variância
        var = sum((x - mean) ** 2 for x in values) / n
        if var == 0:
            return 0.0
            
        # Calcular autocorrelação
        autocorr = sum((values[i] - mean) * (values[i + lag] - mean) 
                      for i in range(n - lag)) / ((n - lag) * var)
                      
        return autocorr
        
    def _calculate_category_regularity(self, meal_history: List[Set[str]]) -> float:
        """
        Calcula regularidade nas categorias alimentares.
        
        Args:
            meal_history: Histórico de categorias alimentares
            
        Returns:
            Índice de regularidade [0-1]
        """
        if len(meal_history) < 3:
            return 0.5
            
        # Contagem de categorias
        category_counts = {}
        total_categories = 0
        
        for meal_categories in meal_history:
            for category in meal_categories:
                if category in category_counts:
                    category_counts[category] += 1
                else:
                    category_counts[category] = 1
                total_categories += 1
                
        if total_categories == 0:
            return 0.5
            
        # Calcular entropia como medida de variabilidade
        entropy = 0
        for count in category_counts.values():
            p = count / total_categories
            entropy -= p * math.log2(p)
            
        # Normalizar entropia para [0-1]
        max_entropy = math.log2(len(category_counts)) if category_counts else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Regularidade é o inverso da entropia normalizada
        return 1.0 - normalized_entropy
        
    def generate_meal_plan(self, 
                          days: int = 1, 
                          target_phi: float = IDEAL_PHI) -> Dict[str, Any]:
        """
        Gera plano alimentar fractal para manter phi alvo.
        
        Args:
            days: Número de dias para o plano
            target_phi: Phi alvo
            
        Returns:
            Plano alimentar com recomendações por refeição
        """
        meal_plan = {
            "target_phi": target_phi,
            "current_phi": self.current_phi,
            "days": []
        }
        
        # Simular evolução de phi ao longo do plano
        simulated_phi = self.current_phi
        
        for day in range(days):
            day_plan = {
                "day": day + 1,
                "meals": []
            }
            
            # Definir tipos de refeição para o dia
            meal_types = ["café da manhã", "almoço", "lanche", "jantar"]
            
            for meal_type in meal_types:
                # Atualizar phi simulado para cada refeição
                self.current_phi = simulated_phi
                
                # Obter recomendações para essa refeição
                recommended, avoided = self.recommend_for_target_phi(target_phi, meal_type)
                
                # Gerar sugestões específicas para o tipo de refeição
                suggestions = self._generate_meal_suggestions(meal_type, recommended, avoided)
                
                # Simular impacto no phi
                phi_impact = self.calculate_phi_impact(recommended, meal_type)
                simulated_phi = max(min(simulated_phi + phi_impact, 1.0), 0.0)
                
                # Adicionar refeição ao plano
                meal_info = {
                    "meal_type": meal_type,
                    "recommended_categories": list(recommended),
                    "avoided_categories": list(avoided),
                    "suggestions": suggestions,
                    "estimated_phi_after": simulated_phi,
                    "phi_impact": phi_impact
                }
                
                day_plan["meals"].append(meal_info)
                
            day_plan["estimated_end_phi"] = simulated_phi
            meal_plan["days"].append(day_plan)
            
        meal_plan["estimated_final_phi"] = simulated_phi
        meal_plan["phi_change"] = simulated_phi - self.current_phi
        
        # Restaurar phi original
        self.current_phi = meal_plan["current_phi"]
        
        return meal_plan
        
    def _generate_meal_suggestions(self, 
                                 meal_type: str, 
                                 recommended_categories: Set[str],
                                 avoided_categories: Set[str]) -> Dict[str, Any]:
        """
        Gera sugestões específicas para uma refeição.
        
        Args:
            meal_type: Tipo de refeição
            recommended_categories: Categorias recomendadas
            avoided_categories: Categorias a evitar
            
        Returns:
            Sugestões para a refeição
        """
        # Mapeamento de opções para diferentes tipos de refeição
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
            }
        }
        
        # Selecionar opções para as categorias recomendadas
        import random
        suggestions = {}
        for category in recommended_categories:
            if category in meal_options.get(meal_type, {}):
                options = meal_options[meal_type][category]
                if options:
                    # Selecionar um ou dois itens aleatoriamente
                    num_items = min(2, len(options))
                    selected = random.sample(options, num_items)
                    suggestions[category] = selected
                    
        # Montar recomendação textual
        main_items = []
        side_items = []
        
        # Priorizar proteínas e carboidratos como itens principais
        if "proteínas" in suggestions:
            main_items.extend(suggestions["proteínas"])
        if "carboidratos" in suggestions and meal_type != "jantar":
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
            },
            "phi_target_alignment": self._calculate_target_alignment(recommended_categories, target_phi=IDEAL_PHI)
        }
        
    def _calculate_target_alignment(self, 
                                  categories: Set[str], 
                                  target_phi: float = IDEAL_PHI) -> float:
        """
        Calcula alinhamento das categorias com o phi alvo.
        
        Args:
            categories: Categorias alimentares
            target_phi: Phi alvo
            
        Returns:
            Valor de alinhamento [-1 a 1]
        """
        # Calcular impacto estimado
        impact = sum(self.category_weights.get(category, 0) for category in categories)
        
        # Calcular gap entre phi atual e alvo
        phi_gap = target_phi - self.current_phi
        
        # Verificar se o impacto está na direção correta
        if (phi_gap > 0 and impact > 0) or (phi_gap < 0 and impact < 0):
            alignment = 1.0
        elif (phi_gap > 0 and impact < 0) or (phi_gap < 0 and impact > 0):
            alignment = -1.0
        else:
            alignment = 0.0
            
        # Modular pelo tamanho do gap e impacto
        return alignment * min(abs(phi_gap), 0.5) * min(abs(impact), 0.5)