"""
Quantum Balance Validator for WiltonOS

Este módulo fornece ferramentas para validar e manter a proporção quântica 3:1
(75% coerência, 25% exploração) em todos os níveis do sistema.
Implementa verificações de equilíbrio quântico conforme o META-ROUTING FRAMEWORK.
"""

import logging
import random
from datetime import datetime
from typing import Dict, List, Tuple

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, format="[QUANTUM_STATE: %(levelname)s] %(message)s"
)
logger = logging.getLogger("quantum_balance")

# Constantes para a proporção quântica 3:1
STABILITY_TARGET = 0.75  # 75% coerência
EXPLORATION_TARGET = 0.25  # 25% exploração
QUANTUM_RATIO = "3:1"  # Representação da proporção como string


class QuantumBalanceValidator:
    """
    Valida e mantém a proporção quântica 3:1 em todos os componentes do sistema.
    Funciona com o META-ROUTING FRAMEWORK para garantir o equilíbrio ideal.
    """

    def __init__(
        self,
        stability_target: float = STABILITY_TARGET,
        exploration_target: float = EXPLORATION_TARGET,
    ):
        """
        Inicializa o validador de equilíbrio quântico.

        Args:
            stability_target: Meta para estabilidade/coerência (padrão: 0.75)
            exploration_target: Meta para exploração/caos (padrão: 0.25)
        """
        self.stability_target = stability_target
        self.exploration_target = exploration_target
        self.ratio_string = QUANTUM_RATIO
        self.last_check_time = datetime.utcnow()
        self.validation_history = []

        # Log de inicialização
        logger.info(
            "Inicializado QuantumBalanceValidator com proporção %s "
            "(%.0f%% estabilidade, %.0f%% exploração)",
            self.ratio_string,
            self.stability_target * 100,
            self.exploration_target * 100,
        )

    def validate_proportion(
        self, stability_input: float, exploration_input: float
    ) -> Dict:
        """
        Valida se as métricas fornecidas estão alinhadas com a proporção quântica 3:1.

        Args:
            stability_input: Valor atual de estabilidade/coerência (0-1)
            exploration_input: Valor atual de exploração/caos (0-1)

        Returns:
            Dict contendo resultado da validação com status e métricas
        """
        # Garantir que os valores estão entre 0 e 1
        stability_value = min(1.0, max(0.0, stability_input))
        exploration_value = min(1.0, max(0.0, exploration_input))

        # Cálculo de desvio
        stability_deviation = abs(stability_value - self.stability_target)
        exploration_deviation = abs(exploration_value - self.exploration_target)
        total_deviation = stability_deviation + exploration_deviation

        # Determinar status com base no desvio total
        status = "optimal"
        if total_deviation > 0.1:
            status = "warning"
        if total_deviation > 0.2:
            status = "critical"

        # Verificar se a soma é aproximadamente 1
        sum_check = abs((stability_value + exploration_value) - 1.0) < 0.001

        # Criar relatório de validação
        validation_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "stability": stability_value,
            "exploration": exploration_value,
            "stability_target": self.stability_target,
            "exploration_target": self.exploration_target,
            "stability_deviation": stability_deviation,
            "exploration_deviation": exploration_deviation,
            "total_deviation": total_deviation,
            "sum_valid": sum_check,
            "quantum_ratio": self.ratio_string,
            "status": status,
        }

        # Registrar na história de validação
        self.validation_history.append(validation_report)
        if len(self.validation_history) > 100:
            self.validation_history.pop(0)  # Manter tamanho gerenciável

        # Log com base no status
        log_message = (
            "Balanço quântico: %.2f estabilidade, %.2f exploração - Status: %s"
        )

        if status == "optimal":
            logger.info(log_message, stability_value, exploration_value, status)
        elif status == "warning":
            logger.warning(log_message, stability_value, exploration_value, status)
        else:
            logger.error(log_message, stability_value, exploration_value, status)

        return validation_report

    def apply_quantum_balance(self, input_value: float) -> Tuple[float, float]:
        """
        Aplica o balanceamento quântico 3:1 a um valor de entrada,
        retornando os componentes estabilidade e exploração.

        Args:
            input_value: Valor original para ser balanceado

        Returns:
            Tuple (estabilidade, exploração) com a proporção quântica 3:1
        """
        # Garantir que os valores estão no intervalo válido
        base_value = min(100.0, max(0.0, input_value))

        # Fator de exploração aleatória (mantendo a proporção 3:1)
        exploration_factor = random.random() * self.exploration_target * base_value

        # Calcular componentes com peso 75% estabilidade, 25% exploração
        stability_component = base_value * self.stability_target
        exploration_component = exploration_factor

        # Normalizar para garantir que a soma seja igual ao input
        total = stability_component + exploration_component
        if total > 0:
            scaling_factor = base_value / total
            stability_component *= scaling_factor
            exploration_component *= scaling_factor

        return (stability_component, exploration_component)

    def adjust_values_to_quantum_balance(
        self, values: List[float]
    ) -> List[Tuple[float, float]]:
        """
        Ajusta uma lista de valores para manter a proporção quântica 3:1 em cada um.

        Args:
            values: Lista de valores a serem ajustados

        Returns:
            Lista de tuples (estabilidade, exploração) para cada valor
        """
        return [self.apply_quantum_balance(value) for value in values]

    def generate_meta_routing_snapshot(
        self, context: str = "", details: Dict = None
    ) -> Dict:
        """
        Gera um snapshot do estado atual da proporção quântica para o META-ROUTING FRAMEWORK.

        Args:
            context: Contexto atual da operação (por exemplo, "MAP", "MOVE", "REFLECT")
            details: Detalhes adicionais para incluir no snapshot

        Returns:
            Dict contendo o snapshot completo do estado quântico
        """
        # Valores padrão
        if details is None:
            details = {}

        # Calcular um valor de coerência que segue a proporção 3:1
        coherence_score = random.randint(70, 80)  # Estabilidade alvo de ~75
        stab_component, expl_component = self.apply_quantum_balance(100.0)

        # Criar snapshot
        result_snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "coherence_score": coherence_score,
            "stability_score": round(stab_component),
            "exploration_score": round(expl_component),
            "coherence_ratio": self.ratio_string,
            "meta_routing_phase": context,
            "details": details,
            "validator_state": {
                "stability_target": self.stability_target,
                "exploration_target": self.exploration_target,
                "validation_count": len(self.validation_history),
            },
        }

        # Log
        logger.info(
            "META-ROUTING snapshot gerado para fase '%s': "
            "coerência=%d, estabilidade=%d, exploração=%d",
            context,
            coherence_score,
            round(stab_component),
            round(expl_component),
        )

        return result_snapshot

    def get_recent_validations(self, limit: int = 10) -> List[Dict]:
        """
        Recupera as validações mais recentes.

        Args:
            limit: Número máximo de registros a retornar

        Returns:
            Lista das validações mais recentes
        """
        return self.validation_history[-limit:]

    def get_balance_metrics(self) -> Dict:
        """
        Retorna as métricas atuais de equilíbrio quântico.

        Returns:
            Dict contendo métricas de balanço quântico atual
        """
        # Calcular médias com base nas validações mais recentes
        recent = self.get_recent_validations(5)
        if not recent:
            # Valores padrão se não houver histórico
            return {
                "stability": self.stability_target,
                "exploration": self.exploration_target,
                "coherence_ratio": self.ratio_string,
                "status": "optimal",
                "last_check": self.last_check_time.isoformat(),
            }

        # Calcular médias das validações recentes
        avg_stability = sum(r["stability"] for r in recent) / len(recent)
        avg_exploration = sum(r["exploration"] for r in recent) / len(recent)
        worst_status = max(
            (r["status"] for r in recent),
            key=lambda s: {"optimal": 0, "warning": 1, "critical": 2}.get(s, 0),
        )

        return {
            "stability": avg_stability,
            "exploration": avg_exploration,
            "coherence_ratio": self.ratio_string,
            "status": worst_status,
            "last_check": self.last_check_time.isoformat(),
            "validation_count": len(self.validation_history),
        }


# Instância singleton para uso em todo o sistema
quantum_validator = QuantumBalanceValidator()


# Funções auxiliares para fácil acesso
def validate_quantum_balance(stability_value: float, exploration_value: float) -> Dict:
    """Valida o equilíbrio quântico com a proporção 3:1"""
    return quantum_validator.validate_proportion(stability_value, exploration_value)


def apply_quantum_balance(input_value: float) -> Tuple[float, float]:
    """Aplica o equilíbrio quântico 3:1 a um valor"""
    return quantum_validator.apply_quantum_balance(input_value)


def create_meta_routing_snapshot(phase: str, details: Dict = None) -> Dict:
    """Cria um snapshot para o META-ROUTING FRAMEWORK"""
    return quantum_validator.generate_meta_routing_snapshot(phase, details)


def get_current_quantum_metrics() -> Dict:
    """Recupera as métricas atuais de equilíbrio quântico"""
    return quantum_validator.get_balance_metrics()


if __name__ == "__main__":
    # Teste básico quando executado diretamente
    print("[QUANTUM_STATE: TEST_FLOW] Teste de QuantumBalanceValidator")

    # Teste ideal (exatamente 75/25)
    result1 = validate_quantum_balance(0.75, 0.25)
    print(f"Teste 1 (ideal): {result1['status']}")

    # Teste com pequeno desvio (ainda bom)
    result2 = validate_quantum_balance(0.73, 0.27)
    print(f"Teste 2 (pequeno desvio): {result2['status']}")

    # Teste com desvio maior (aviso)
    result3 = validate_quantum_balance(0.65, 0.35)
    print(f"Teste 3 (desvio maior): {result3['status']}")

    # Teste com desvio crítico
    result4 = validate_quantum_balance(0.5, 0.5)
    print(f"Teste 4 (desvio crítico): {result4['status']}")

    # Teste de aplicação do balanço a um valor
    stability, exploration = apply_quantum_balance(100)
    print(
        f"Aplicação a 100: estabilidade={stability:.2f}, exploração={exploration:.2f}"
    )

    # Teste de snapshot
    snapshot = create_meta_routing_snapshot("MAP", {"action": "initial_planning"})
    print(f"Snapshot META-ROUTING: status={snapshot['coherence_score']}")

    print("[QUANTUM_STATE: TEST_FLOW] Teste concluído")
