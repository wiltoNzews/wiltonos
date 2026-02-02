#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificador de Balanço Quântico 3:1

Este módulo contém funções para verificar e validar o balanço quântico
de proporção 3:1 (75% estabilidade, 25% exploração) em todos os componentes
do sistema META-ROUTING FRAMEWORK.

O verificador conecta-se ao servidor WebSocket do META-ROUTING FRAMEWORK
para obter métricas de coerência em tempo real e avaliar se o sistema está
mantendo o equilíbrio adequado. Ele pode ser executado como ferramenta de 
linha de comando ou importado e utilizado programaticamente.

Exemplos:
    # Uso via linha de comando
    python -m wilton_core.qctf.balance_verifier --duration 60 --verbose
    
    # Uso programático
    from wilton_core.qctf.balance_verifier import QuantumBalanceVerifier
    
    async def verify_balance():
        verifier = QuantumBalanceVerifier()
        await verifier.connect()
        await verifier.listen_for_updates(duration_seconds=30)
        is_balanced, status = verifier.verify_balance()
        print(f"Sistema está equilibrado: {is_balanced} - {status}")
        verifier.print_detailed_report()
        await verifier.disconnect()
"""

import os
import sys
import json
import asyncio
import logging
import argparse
import websockets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

# Adicionar diretório raiz ao path para importações relativas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from wilton_core.websocket.config import (
        DEFAULT_STABILITY,
        DEFAULT_EXPLORATION,
        WS_HOST,
        WS_PORT,
        WS_PATH,
        get_websocket_url,
        get_status_for_ratio,
    )
except ImportError:
    # Valores padrão caso o módulo de configuração não esteja disponível
    DEFAULT_STABILITY = 0.75
    DEFAULT_EXPLORATION = 0.25
    WS_HOST = "0.0.0.0"
    WS_PORT = 8765
    WS_PATH = "/ws"

    def get_websocket_url():
        return f"ws://{WS_HOST}:{WS_PORT}{WS_PATH}"

    def get_status_for_ratio(ratio):
        if 2.9 <= ratio <= 3.1:
            return "optimal"
        elif 2.5 <= ratio <= 3.5:
            return "suboptimal"
        else:
            return "critical"


# Configuração de logging
logging.basicConfig(
    level=logging.INFO, format="[BALANCE_VERIFIER: %(levelname)s] %(message)s"
)
logger = logging.getLogger("balance_verifier")

# Constantes
TARGET_RATIO = 3.0
VERIFICATION_TIMEOUT = 30  # Segundos para considerar timeout
MAXIMUM_DEVIATION = 0.5  # Desvio máximo aceitável do alvo (3.0 ± 0.5)
OPTIMAL_DEVIATION = 0.1  # Desvio para status ótimo (3.0 ± 0.1)


class QuantumBalanceVerifier:
    """
    Verificador do balanço quântico na proporção 3:1
    
    Esta classe implementa a lógica de verificação do balanço quântico,
    conectando-se ao WebSocket do META-ROUTING FRAMEWORK para coletar
    métricas de estabilidade e exploração em tempo real. Ela analisa
    estas métricas para determinar se o sistema está mantendo o balanço
    quântico adequado (proporção 3:1).
    
    O verificador monitora:
    - A proporção estabilidade/exploração atual e histórica
    - A percentagem de amostras com proporção ótima (3.0 ± 0.1)
    - Desvios significativos da proporção alvo
    - Distribuição de estados (ótimo, subótimo, crítico)
    
    Exemplos:
        # Verificação básica por tempo determinado
        verifier = QuantumBalanceVerifier()
        
        # Usar o event loop para operações assíncronas
        import asyncio
        
        async def check_balance():
            await verifier.connect()
            success = await verifier.listen_for_updates(duration_seconds=10)
            
            if success:
                balanced, status = verifier.verify_balance()
                print(f"Sistema equilibrado: {balanced}, Status: {status}")
                
                # Mostrar relatório detalhado
                verifier.print_detailed_report()
            else:
                print("Falha ao obter métricas do sistema")
                
            await verifier.disconnect()
            
        asyncio.run(check_balance())
    """

    def __init__(self, ws_url: Optional[str] = None):
        """
        Inicializa o verificador com a URL do WebSocket.

        Args:
            ws_url: URL do WebSocket (opcional, usa valor padrão se não informada)
        """
        self.ws_url = ws_url or get_websocket_url()
        self.last_heartbeat: Optional[str] = None
        self.balance_metrics: List[Dict[str, Any]] = []
        self.phases_history: List[Dict[str, Any]] = []
        self.is_connected = False
        self.connection_time: Optional[datetime] = None

    async def connect(self) -> bool:
        """
        Conecta ao servidor WebSocket e recebe a mensagem de boas-vindas.

        Returns:
            bool: True se a conexão foi estabelecida com sucesso
        """
        try:
            self.websocket = await websockets.connect(self.ws_url, ping_interval=20)
            self.is_connected = True
            self.connection_time = datetime.now()

            # Receber a mensagem de boas-vindas
            welcome = await self.websocket.recv()
            welcome_data = json.loads(welcome)

            if welcome_data["type"] == "welcome":
                # Extrair métricas de balanço da mensagem de boas-vindas
                if "data" in welcome_data and "quantum_balance" in welcome_data["data"]:
                    qb = welcome_data["data"]["quantum_balance"]
                    self.balance_metrics.append(
                        {
                            "timestamp": welcome_data["timestamp"],
                            "stability": qb.get("stability", DEFAULT_STABILITY),
                            "exploration": qb.get("exploration", DEFAULT_EXPLORATION),
                            "status": qb.get("status", "unknown"),
                            "source": "welcome",
                        }
                    )

                # Extrair histórico de fases
                if (
                    "data" in welcome_data
                    and "meta_routing_state" in welcome_data["data"]
                ):
                    mrs = welcome_data["data"]["meta_routing_state"]
                    self.phases_history = mrs.get("phases_history", [])

            logger.info(f"Conectado ao servidor WebSocket: {self.ws_url}")
            return True

        except Exception as e:
            logger.error(f"Erro ao conectar ao servidor WebSocket: {e}")
            self.is_connected = False
            return False

    async def disconnect(self) -> None:
        """Desconecta do servidor WebSocket"""
        if self.is_connected and hasattr(self, 'websocket'):
            await self.websocket.close()
            self.is_connected = False
            logger.info("Desconectado do servidor WebSocket")

    async def listen_for_updates(self, duration_seconds: int = 30) -> bool:
        """
        Escuta por atualizações do WebSocket por um período determinado.

        Args:
            duration_seconds: Tempo em segundos para escutar atualizações

        Returns:
            bool: True se recebeu atualizações com sucesso
        """
        if not self.is_connected:
            logger.error("Não conectado ao servidor WebSocket")
            return False

        end_time = datetime.now() + timedelta(seconds=duration_seconds)
        heartbeats = 0
        balance_updates = 0

        try:
            while datetime.now() < end_time:
                # Definir timeout para não bloquear indefinidamente
                remaining = (end_time - datetime.now()).total_seconds()

                if remaining <= 0:
                    break

                try:
                    # Aguardar mensagem com timeout
                    message = await asyncio.wait_for(
                        self.websocket.recv(), timeout=min(remaining, 5)
                    )
                    data = json.loads(message)

                    # Processar mensagem de acordo com o tipo
                    if data["type"] == "heartbeat":
                        self.last_heartbeat = data["timestamp"]
                        heartbeats += 1

                    elif data["type"] == "quantum_balance_update":
                        balance_updates += 1
                        if "data" in data:
                            qb = data["data"]
                            self.balance_metrics.append(
                                {
                                    "timestamp": data["timestamp"],
                                    "stability": qb.get("stability", DEFAULT_STABILITY),
                                    "exploration": qb.get(
                                        "exploration", DEFAULT_EXPLORATION
                                    ),
                                    "status": qb.get("status", "unknown"),
                                    "meta_routing_phase": qb.get(
                                        "meta_routing_phase", "unknown"
                                    ),
                                    "source": "update",
                                }
                            )

                    elif data["type"] == "meta_routing_phase_updated":
                        if "data" in data and "phase" in data["data"]:
                            self.phases_history.append(
                                {
                                    "phase": data["data"]["phase"],
                                    "timestamp": data["timestamp"],
                                    "details": data["data"].get("details", {}),
                                }
                            )

                except asyncio.TimeoutError:
                    # Timeout é esperado, continuar esperando
                    continue
                except Exception as e:
                    logger.error(f"Erro ao receber mensagem: {e}")
                    break

            logger.info(
                f"Escuta finalizada após {duration_seconds}s: "
                + f"{heartbeats} heartbeats, {balance_updates} atualizações de balanço"
            )
            return heartbeats > 0 or balance_updates > 0

        except Exception as e:
            logger.error(f"Erro durante escuta por atualizações: {e}")
            return False

    def calculate_ratio_statistics(self) -> Dict[str, Any]:
        """
        Calcula estatísticas da proporção quântica com base nas métricas recebidas.

        Returns:
            Dict[str, Any]: Estatísticas calculadas
        """
        if not self.balance_metrics:
            return {
                "count": 0,
                "average_ratio": 0,
                "min_ratio": 0,
                "max_ratio": 0,
                "optimal_count": 0,
                "optimal_percentage": 0,
                "suboptimal_count": 0,
                "critical_count": 0,
                "overall_status": "unknown",
            }

        ratios = []
        optimal_count = 0
        suboptimal_count = 0
        critical_count = 0

        for metric in self.balance_metrics:
            stability = metric["stability"]
            exploration = metric["exploration"]

            if exploration > 0:
                ratio = stability / exploration
                ratios.append(ratio)

                status = get_status_for_ratio(ratio)
                if status == "optimal":
                    optimal_count += 1
                elif status == "suboptimal":
                    suboptimal_count += 1
                else:
                    critical_count += 1

        if not ratios:
            return {
                "count": len(self.balance_metrics),
                "average_ratio": 0,
                "min_ratio": 0,
                "max_ratio": 0,
                "optimal_count": 0,
                "optimal_percentage": 0,
                "suboptimal_count": 0,
                "critical_count": 0,
                "overall_status": "unknown",
            }

        avg_ratio = sum(ratios) / len(ratios)
        optimal_percentage = (optimal_count / len(ratios)) * 100 if ratios else 0

        # Determinar status geral
        if optimal_percentage >= 80:
            overall_status = "optimal"
        elif optimal_percentage >= 50:
            overall_status = "suboptimal"
        else:
            overall_status = "critical"

        return {
            "count": len(ratios),
            "average_ratio": avg_ratio,
            "min_ratio": min(ratios),
            "max_ratio": max(ratios),
            "optimal_count": optimal_count,
            "optimal_percentage": optimal_percentage,
            "suboptimal_count": suboptimal_count,
            "critical_count": critical_count,
            "overall_status": overall_status,
        }

    def verify_balance(self) -> Tuple[bool, str]:
        """
        Verifica se o balanço quântico está conforme esperado.

        Returns:
            Tuple[bool, str]: (resultado da verificação, mensagem de status)
        """
        stats = self.calculate_ratio_statistics()

        if stats["count"] == 0:
            return False, "Nenhuma métrica de balanço quântico recebida"

        avg_ratio = stats["average_ratio"]
        ratio_deviation = abs(avg_ratio - TARGET_RATIO)

        # Verificar desvio em relação ao alvo
        if ratio_deviation <= OPTIMAL_DEVIATION:
            result = True
            message = f"Balanço quântico ÓTIMO: {avg_ratio:.2f}:1 (desvio: {ratio_deviation:.2f})"
        elif ratio_deviation <= MAXIMUM_DEVIATION:
            result = True
            message = f"Balanço quântico ACEITÁVEL: {avg_ratio:.2f}:1 (desvio: {ratio_deviation:.2f})"
        else:
            result = False
            message = f"Balanço quântico CRÍTICO: {avg_ratio:.2f}:1 (desvio: {ratio_deviation:.2f})"

        # Adicionar detalhes à mensagem
        message += (
            f"\n - {stats['optimal_percentage']:.1f}% das medições no range ótimo"
        )
        message += (
            f"\n - Min: {stats['min_ratio']:.2f}:1, Max: {stats['max_ratio']:.2f}:1"
        )

        return result, message

    def print_detailed_report(self) -> None:
        """
        Imprime um relatório detalhado da verificação do balanço quântico
        
        Esta função analisa as métricas coletadas e gera um relatório formatado
        exibindo:
        - Status da conexão com o servidor
        - Estatísticas de estabilidade e exploração
        - Distribuição das proporções observadas
        - Média, mínimo e máximo da proporção
        - Percentuais de amostras nas categorias: ótimo, subótimo e crítico
        - Recomendações se o balanço não estiver adequado
        
        O relatório é impresso diretamente na saída padrão e é formatado
        para facilitar a leitura e análise dos dados.
        """
        stats = self.calculate_ratio_statistics()

        print("\n" + "=" * 70)
        print(" RELATÓRIO DE VERIFICAÇÃO DO BALANÇO QUÂNTICO 3:1")
        print("=" * 70)

        if not self.is_connected:
            print("\n❌ ERRO: Não foi possível conectar ao servidor WebSocket")
            return

        if not self.last_heartbeat:
            print("\n❌ ERRO: Nenhum heartbeat recebido")
        else:
            print(f"\n✓ Último heartbeat: {self.last_heartbeat}")

        if stats["count"] == 0:
            print("\n❌ ERRO: Nenhuma métrica de balanço quântico recebida")
            return

        status_icon = {
            "optimal": "✅",
            "suboptimal": "⚠️",
            "critical": "❌",
            "unknown": "❓",
        }.get(stats["overall_status"], "❓")

        print(
            f"\n{status_icon} Status geral do balanço: {stats['overall_status'].upper()}"
        )
        print(f"\nEstatísticas da proporção (alvo: {TARGET_RATIO}:1):")
        print(f" - Número de medições: {stats['count']}")
        print(f" - Proporção média: {stats['average_ratio']:.3f}:1")
        print(f" - Desvio do alvo: {abs(stats['average_ratio'] - TARGET_RATIO):.3f}")
        print(
            f" - Range observado: {stats['min_ratio']:.2f}:1 - {stats['max_ratio']:.2f}:1"
        )

        print("\nDistribuição por status:")
        print(
            f" - Optimal: {stats['optimal_count']} ({stats['optimal_percentage']:.1f}%)"
        )
        print(
            f" - Suboptimal: {stats['suboptimal_count']} "
            + f"({(stats['suboptimal_count'] / stats['count'] * 100):.1f}%)"
        )
        print(
            f" - Critical: {stats['critical_count']} "
            + f"({(stats['critical_count'] / stats['count'] * 100):.1f}%)"
        )

        print("\nHistórico de fases:")
        if not self.phases_history:
            print(" - Nenhuma fase registrada")
        else:
            for i, phase in enumerate(self.phases_history[-5:]):
                phase_name = phase.get("phase", "UNKNOWN")
                timestamp = phase.get("timestamp", "?")
                print(f" {i+1}. {phase_name} em {timestamp}")

        # Resultado final da verificação
        result, message = self.verify_balance()
        print("\nCONCLUSÃO:")
        passed_text = "✅ PASSOU" if result else "❌ FALHOU"
        first_line = message.split("\n")[0]
        print(f"{passed_text}: {first_line}")
        print("=" * 70)


async def main() -> int:
    """
    Função principal do verificador de balanço quântico
    
    Implementa a lógica de linha de comando para o verificador, processando 
    argumentos, conectando-se ao servidor WebSocket, coletando métricas e 
    gerando o relatório de verificação.
    
    Códigos de retorno:
        0: Verificação bem-sucedida (balanço quântico adequado)
        1: Verificação falhou (balanço quântico inadequado ou erro)
        2: Verificação interrompida pelo usuário
    
    Exemplos de uso:
        # Verificar usando configurações padrão
        python -m wilton_core.qctf.balance_verifier
        
        # Verificar por 60 segundos
        python -m wilton_core.qctf.balance_verifier --duration 60
        
        # Conectar a um servidor WebSocket específico
        python -m wilton_core.qctf.balance_verifier --host 10.0.0.1 --port 8765
    
    Returns:
        int: Código de retorno indicando o resultado da verificação
    """
    parser = argparse.ArgumentParser(description="Verificador de Balanço Quântico 3:1")
    parser.add_argument("--host", default=WS_HOST, help="Host do servidor WebSocket")
    parser.add_argument(
        "--port", type=int, default=WS_PORT, help="Porta do servidor WebSocket"
    )
    parser.add_argument("--path", default=WS_PATH, help="Caminho do WebSocket")
    parser.add_argument(
        "--duration", type=int, default=30, help="Duração da verificação em segundos"
    )
    args = parser.parse_args()

    ws_url = f"ws://{args.host}:{args.port}{args.path}"
    verifier = QuantumBalanceVerifier(ws_url)

    try:
        # Conectar ao servidor WebSocket
        if not await verifier.connect():
            logger.error("Falha ao conectar ao servidor WebSocket")
            return 1

        # Escutar por atualizações durante o período especificado
        await verifier.listen_for_updates(args.duration)

        # Imprimir relatório detalhado
        verifier.print_detailed_report()

        # Verificar o balanço
        result, _ = verifier.verify_balance()

        # Desconectar
        await verifier.disconnect()

        return 0 if result else 1

    except KeyboardInterrupt:
        print("\nVerificação interrompida pelo usuário")
        await verifier.disconnect()
        return 2
    except Exception as e:
        logger.error(f"Erro durante a verificação: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperação interrompida pelo usuário")
        sys.exit(2)
