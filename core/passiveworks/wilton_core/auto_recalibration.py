#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-Recalibration Service (ARS) - Reequilíbrio Quântico

Este módulo implementa o serviço de auto-recalibração que reage
aos desvios da proporção quântica 3:1 (75% estabilidade, 25% exploração)
detectados pelo Quantum Gauge, ajustando parâmetros do sistema para
restaurar o equilíbrio adequado.

O ARS monitora continuamente o equilíbrio e, ao detectar desvios significativos,
inicia um processo de recalibração que zera temporariamente o orçamento de exploração
para permitir que o sistema retorne ao estado de equilíbrio ideal.

Exemplos de uso:
    # Inicialização básica
    from wilton_core.auto_recalibration import get_auto_recalibration_service
    
    # Obter a instância singleton do serviço
    ars = get_auto_recalibration_service()
    
    # Iniciar o serviço em background
    ars.start()
    
    # Uso com callbacks personalizados
    def on_budget_change(budget_level):
        print(f"Orçamento de exploração alterado para: {budget_level:.2f}")
        
    def on_recalibration_start(event):
        print(f"Iniciando recalibração devido a desvio: {event.trigger_event.direction}")
        
    ars.set_exploration_budget_callback(on_budget_change)
    ars.set_notification_callbacks(on_start=on_recalibration_start)

Classes:
    RecalibrationEvent: Representa um evento de recalibração
    AutoRecalibrationService: Implementa o serviço de auto-recalibração
"""

import os
import time
import json
import asyncio
import logging
import threading
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta

# Importar o Quantum Gauge
from wilton_core.quantum_gauge import Gauge, GaugeState, get_gauge, DriftEvent

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="[ARS: %(levelname)s] %(message)s")
logger = logging.getLogger("auto_recalibration")

# Constantes para configuração do ARS
DEFAULT_RECALIBRATION_DURATION = 30  # Duração em segundos da recalibração
DEFAULT_CHECK_INTERVAL = 1  # Intervalo em segundos para verificar o gauge


class RecalibrationEvent:
    """
    Evento de recalibração quântica
    
    Esta classe representa um evento de recalibração iniciado em resposta a um desvio 
    detectado na proporção quântica 3:1. Ela mantém o estado da recalibração
    incluindo os valores pré e pós-recalibração, duração e resultado.
    
    Exemplos:
        # Criar um evento a partir de um drift detectado
        from wilton_core.quantum_gauge import DriftEvent
        from datetime import datetime
        
        drift = DriftEvent(timestamp=datetime.now(), current_ratio=2.0, 
                          avg_ratio=2.2, consecutive_samples=3, direction="low")
        
        event = RecalibrationEvent(drift, datetime.now(), target_duration=30)
        
        # Verificar progresso
        print(f"Progresso: {event.progress_percentage()}%")
        
        # Completar evento após recalibração
        event.complete(0.75)  # Estabilidade após recalibração
        
        # Verificar se foi bem-sucedido
        if event.metrics["success"]:
            print("Recalibração bem-sucedida!")
    """

    def __init__(
        self,
        trigger_event: DriftEvent,
        start_time: datetime,
        target_duration: int = DEFAULT_RECALIBRATION_DURATION,
    ):
        """
        Inicializa um evento de recalibração

        Args:
            trigger_event: Evento de drift que disparou a recalibração
            start_time: Timestamp de início da recalibração
            target_duration: Duração alvo da recalibração em segundos
        """
        self.trigger_event = trigger_event
        self.start_time = start_time
        self.target_duration = target_duration
        self.target_end_time = start_time + timedelta(seconds=target_duration)
        self.actual_end_time: Optional[datetime] = None
        self.status = "in_progress"  # in_progress, completed, cancelled
        self.metrics: Dict[str, Any] = {
            "pre_stability": trigger_event.current_ratio
            / (1 + trigger_event.current_ratio),
            "pre_ratio": trigger_event.current_ratio,
            "post_stability": None,
            "post_ratio": None,
            "duration_seconds": None,
            "success": None,
        }

    def complete(self, current_stability: float) -> None:
        """
        Marca a recalibração como concluída

        Args:
            current_stability: Valor de estabilidade após a recalibração
        """
        self.actual_end_time = datetime.now()
        self.status = "completed"

        # Calcular métricas
        self.metrics["post_stability"] = current_stability
        self.metrics["post_ratio"] = current_stability / (1 - current_stability)
        self.metrics["duration_seconds"] = (
            self.actual_end_time - self.start_time
        ).total_seconds()

        # Verificar se a recalibração teve sucesso
        pre_ratio = self.metrics["pre_ratio"]
        post_ratio = self.metrics["post_ratio"]

        # Define sucesso se a razão está mais próxima de 3.0 do que antes
        self.metrics["success"] = abs(post_ratio - 3.0) < abs(pre_ratio - 3.0)

    def cancel(self) -> None:
        """Marca a recalibração como cancelada"""
        self.actual_end_time = datetime.now()
        self.status = "cancelled"
        self.metrics["duration_seconds"] = (
            self.actual_end_time - self.start_time
        ).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """
        Converte o evento para dicionário

        Returns:
            Dict[str, Any]: Representação em dicionário do evento
        """
        return {
            "trigger": self.trigger_event.to_dict(),
            "start_time": self.start_time.isoformat(),
            "target_duration": self.target_duration,
            "target_end_time": self.target_end_time.isoformat(),
            "actual_end_time": (
                self.actual_end_time.isoformat() if self.actual_end_time else None
            ),
            "status": self.status,
            "metrics": self.metrics,
        }

    def time_remaining(self) -> int:
        """
        Retorna o tempo restante da recalibração em segundos

        Returns:
            int: Segundos restantes (0 se já concluído)
        """
        if self.status != "in_progress":
            return 0

        remaining = (self.target_end_time - datetime.now()).total_seconds()
        return max(0, int(remaining))

    def progress_percentage(self) -> int:
        """
        Retorna o progresso da recalibração em porcentagem

        Returns:
            int: Porcentagem de progresso (0-100)
        """
        if self.status != "in_progress":
            return 100

        elapsed = (datetime.now() - self.start_time).total_seconds()
        percentage = (elapsed / self.target_duration) * 100
        return min(100, int(percentage))


class AutoRecalibrationService:
    """
    Serviço de Auto-Recalibração (ARS)

    Monitora o Quantum Gauge e inicia procedimentos de recalibração
    quando são detectados desvios na proporção quântica 3:1 (75% estabilidade,
    25% exploração). Implementa o mecanismo de auto-cura do sistema para
    manter o equilíbrio quântico mesmo em condições de stress.
    
    O serviço opera em background usando uma thread daemon, verificando
    periodicamente o estado do Quantum Gauge. Quando um desvio (drift)
    é detectado, o serviço inicia um processo de recalibração que:
    
    1. Zera temporariamente o orçamento de exploração
    2. Aguarda um período definido (recalibration_duration)
    3. Restaura o orçamento de exploração
    4. Avalia a eficácia da recalibração
    
    Exemplos:
        # Uso básico
        ars = AutoRecalibrationService()
        ars.start()
        
        # Configuração personalizada
        custom_gauge = Gauge(stability_target=0.8)
        ars = AutoRecalibrationService(
            gauge=custom_gauge,
            recalibration_duration=60,  # 1 minuto
            check_interval=2            # Verificar a cada 2 segundos
        )
        ars.start()
        
        # Verificar status atual
        status = ars.get_status()
        print(f"Em recalibração: {status['current_recalibration'] is not None}")
        print(f"Total de recalibrações: {status['metrics']['recalibrations_started']}")
        
        # Parar o serviço quando não for mais necessário
        ars.stop()
    """

    def __init__(
        self,
        gauge: Optional[Gauge] = None,
        recalibration_duration: int = DEFAULT_RECALIBRATION_DURATION,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
    ):
        """
        Inicializa o serviço de auto-recalibração

        Args:
            gauge: Instância do Quantum Gauge (opcional, usa singleton se None)
            recalibration_duration: Duração da recalibração em segundos
            check_interval: Intervalo de verificação do gauge em segundos
        """
        self.gauge = gauge or get_gauge()
        self.recalibration_duration = recalibration_duration
        self.check_interval = check_interval

        # Estado interno
        self.running = False
        self.paused = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Eventos de recalibração
        self.current_recalibration: Optional[RecalibrationEvent] = None
        self.past_recalibrations: List[RecalibrationEvent] = []

        # Callback para modificar o orçamento de exploração
        self.exploration_budget_callback: Optional[Callable[[float], None]] = None

        # Callbacks para notificações
        self.on_recalibration_start: Optional[Callable[[RecalibrationEvent], None]] = (
            None
        )
        self.on_recalibration_progress: Optional[
            Callable[[RecalibrationEvent], None]
        ] = None
        self.on_recalibration_end: Optional[Callable[[RecalibrationEvent], None]] = None

        # Métricas
        self.metrics = {
            "recalibrations_started": 0,
            "recalibrations_completed": 0,
            "recalibrations_cancelled": 0,
            "total_recalibration_seconds": 0,
            "last_recalibration_timestamp": None,
        }

        logger.info(
            f"Auto-Recalibration Service iniciado - "
            f"Duração de recalibração: {recalibration_duration}s, "
            f"Intervalo de verificação: {check_interval}s"
        )

    def start(self) -> None:
        """Inicia o serviço em background"""
        if self.running:
            logger.warning("O serviço já está em execução")
            return

        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Auto-Recalibration Service iniciado em background")

    def stop(self) -> None:
        """Para o serviço"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        if self.current_recalibration:
            self.current_recalibration.cancel()
            if self.on_recalibration_end:
                self.on_recalibration_end(self.current_recalibration)
            self.current_recalibration = None

        logger.info("Auto-Recalibration Service parado")

    def pause(self) -> None:
        """Pausa o serviço temporariamente"""
        self.paused = True
        logger.info("Auto-Recalibration Service pausado")

    def resume(self) -> None:
        """Retoma o serviço após pausa"""
        self.paused = False
        logger.info("Auto-Recalibration Service retomado")

    def set_exploration_budget_callback(
        self, callback: Callable[[float], None]
    ) -> None:
        """
        Define o callback para controlar o orçamento de exploração

        Args:
            callback: Função que recebe o nível de orçamento (0.0 - 1.0)
        """
        self.exploration_budget_callback = callback

    def set_notification_callbacks(
        self,
        on_start: Optional[Callable[[RecalibrationEvent], None]] = None,
        on_progress: Optional[Callable[[RecalibrationEvent], None]] = None,
        on_end: Optional[Callable[[RecalibrationEvent], None]] = None,
    ) -> None:
        """
        Define callbacks para notificações de eventos

        Args:
            on_start: Callback chamado ao iniciar uma recalibração
            on_progress: Callback chamado durante a recalibração
            on_end: Callback chamado ao finalizar uma recalibração
        """
        self.on_recalibration_start = on_start
        self.on_recalibration_progress = on_progress
        self.on_recalibration_end = on_end

    def get_status(self) -> Dict[str, Any]:
        """
        Retorna o status atual do serviço

        Returns:
            Dict[str, Any]: Status atual do serviço
        """
        with self._lock:
            return {
                "running": self.running,
                "paused": self.paused,
                "current_recalibration": (
                    self.current_recalibration.to_dict()
                    if self.current_recalibration
                    else None
                ),
                "metrics": self.metrics,
                "past_recalibrations_count": len(self.past_recalibrations),
                "gauge_state": self.gauge.state,
            }

    def _run_loop(self) -> None:
        """Loop principal do serviço"""
        while self.running:
            try:
                if not self.paused:
                    self._check_gauge()

                # Atualizar recalibração em andamento
                self._update_current_recalibration()

                # Pausa até próxima verificação
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Erro no loop do ARS: {e}")
                time.sleep(1)  # Evitar ciclo de erro contínuo

    def _check_gauge(self) -> None:
        """Verifica o estado do gauge e reage a desvios"""
        with self._lock:
            # Verificar se já existe recalibração em andamento
            if self.current_recalibration:
                return

            # Verificar se o gauge está em estado de drift
            if self.gauge.state == GaugeState.DRIFT and self.gauge.last_drift_event:
                self._start_recalibration(self.gauge.last_drift_event)

    def _start_recalibration(self, drift_event: DriftEvent) -> None:
        """
        Inicia o processo de recalibração

        Args:
            drift_event: Evento de drift que disparou a recalibração
        """
        with self._lock:
            logger.info(
                f"Iniciando recalibração - Deriva detectada: "
                f"estabilidade={drift_event.current_ratio / (1 + drift_event.current_ratio):.3f}, "
                f"razão={drift_event.current_ratio:.2f}:1"
            )

            # Iniciar processo de recalibração
            self.current_recalibration = RecalibrationEvent(
                drift_event, datetime.now(), self.recalibration_duration
            )

            # Atualizar métricas
            self.metrics["recalibrations_started"] += 1
            self.metrics["last_recalibration_timestamp"] = datetime.now().isoformat()

            # Atualizar estado do gauge
            self.gauge.start_recalibration()

            # Pausar orçamento de exploração
            if self.exploration_budget_callback:
                self.exploration_budget_callback(0.0)
                logger.info("Orçamento de exploração zerado durante recalibração")

            # Notificar início
            if self.on_recalibration_start:
                self.on_recalibration_start(self.current_recalibration)

    def _update_current_recalibration(self) -> None:
        """Atualiza e verifica o status da recalibração atual"""
        with self._lock:
            if not self.current_recalibration:
                return

            # Verificar progresso
            progress = self.current_recalibration.progress_percentage()

            # Notificar progresso a cada 10%
            if progress % 10 == 0 and self.on_recalibration_progress:
                self.on_recalibration_progress(self.current_recalibration)

            # Verificar se deve finalizar
            if self.current_recalibration.time_remaining() <= 0:
                self._complete_recalibration()

    def _complete_recalibration(self) -> None:
        """Completa o processo de recalibração"""
        with self._lock:
            if not self.current_recalibration:
                return

            # Obter estabilidade atual
            stability = (
                self.gauge.samples[-1]
                if self.gauge.samples
                else self.gauge.stability_target
            )

            # Marcar como concluída
            self.current_recalibration.complete(stability)

            # Atualizar métricas
            self.metrics["recalibrations_completed"] += 1
            self.metrics["total_recalibration_seconds"] += (
                self.current_recalibration.metrics["duration_seconds"] or 0
            )

            # Restaurar estado do gauge
            self.gauge.end_recalibration()

            # Restaurar orçamento de exploração
            if self.exploration_budget_callback:
                self.exploration_budget_callback(1.0)
                logger.info("Orçamento de exploração restaurado após recalibração")

            # Registrar resultado
            success = self.current_recalibration.metrics["success"]
            logger.info(
                f"Recalibração finalizada - "
                f"Resultado: {'SUCESSO' if success else 'SUBÓTIMO'}, "
                f"Nova estabilidade: {stability:.3f}, "
                f"Nova razão: {stability / (1-stability) if (1-stability) > 0 else float('inf'):.2f}:1"
            )

            # Notificar conclusão
            if self.on_recalibration_end:
                self.on_recalibration_end(self.current_recalibration)

            # Salvar no histórico e limpar atual
            self.past_recalibrations.append(self.current_recalibration)
            self.current_recalibration = None


# Instância singleton do serviço
_ars_instance = None


def get_auto_recalibration_service() -> AutoRecalibrationService:
    """
    Retorna a instância singleton do serviço de auto-recalibração
    
    Esta função garante que apenas uma instância do Auto-Recalibration Service
    exista no sistema, seguindo o padrão Singleton. Na primeira chamada, ela cria
    a instância; nas chamadas seguintes, retorna a mesma instância.
    
    Exemplo:
        # Obter a instância singleton em qualquer parte do código
        from wilton_core.auto_recalibration import get_auto_recalibration_service
        
        ars = get_auto_recalibration_service()
        
        # Configurar e usar a instância
        ars.set_exploration_budget_callback(my_callback)
        ars.start()
    
    Returns:
        AutoRecalibrationService: Instância singleton do serviço
    """
    global _ars_instance
    if _ars_instance is None:
        _ars_instance = AutoRecalibrationService()
    return _ars_instance


# Função para testes
def main():
    """
    Função principal para testes do serviço de auto-recalibração
    
    Esta função demonstra a utilização do Auto-Recalibration Service
    em um cenário de teste, incluindo:
    
    1. Configuração dos callbacks para orçamento de exploração e notificações
    2. Simulação de amostras de estabilidade normais
    3. Simulação de um desvio (drift) para disparar uma recalibração
    4. Monitoramento do progresso da recalibração
    5. Verificação do status final do serviço
    
    Para executar este teste:
    ```python
    python -m wilton_core.auto_recalibration
    ```
    """
    from wilton_core.quantum_gauge import Gauge

    # Criar gauge e ARS
    gauge = Gauge()
    ars = AutoRecalibrationService(gauge=gauge)

    # Definir callback de exemplo
    def exploration_budget_callback(budget):
        print(f"Orçamento de exploração alterado para: {budget:.2f}")

    ars.set_exploration_budget_callback(exploration_budget_callback)

    # Definir callbacks de notificação
    def on_recalibration_start(event):
        print(f"\n⚙️ RECALIBRAÇÃO INICIADA - Trigger: {event.trigger_event.direction}")

    def on_recalibration_progress(event):
        print(f"⏳ Progresso: {event.progress_percentage()}%")

    def on_recalibration_end(event):
        print(
            f"✓ RECALIBRAÇÃO FINALIZADA - Sucesso: {'Sim' if event.metrics['success'] else 'Não'}"
        )

    ars.set_notification_callbacks(
        on_start=on_recalibration_start,
        on_progress=on_recalibration_progress,
        on_end=on_recalibration_end,
    )

    # Iniciar serviço em modo teste (duração curta)
    ars.recalibration_duration = 5  # 5 segundos para teste
    ars.start()

    print("\nSimulando amostras para teste do Auto-Recalibration Service...\n")

    try:
        # Amostras normais
        for i in range(3):
            value = 0.75 + (i * 0.01)
            state, event = gauge.update(value)
            print(f"Amostra {i+1}: Estabilidade={value:.3f}, Estado={state}")
            time.sleep(0.5)

        # Simular drift para baixo (deve disparar recalibração)
        print("\nSimulando drift para baixo (estabilidade baixa)...\n")
        for i in range(5):
            value = 0.5 - (i * 0.05)
            state, event = gauge.update(value)
            print(f"Amostra {i+4}: Estabilidade={value:.3f}, Estado={state}")
            time.sleep(0.5)

        # Aguardar a recalibração completa
        print("\nAguardando recalibração (5s)...\n")
        time.sleep(7)

        # Verificar status
        status = ars.get_status()
        print("\nStatus do ARS:")
        print(f"  Estado: {'Ativo' if status['running'] else 'Inativo'}")
        print(
            f"  Recalibrações iniciadas: {status['metrics']['recalibrations_started']}"
        )
        print(
            f"  Recalibrações concluídas: {status['metrics']['recalibrations_completed']}"
        )

        # Parar serviço
        ars.stop()
        print("\nServiço parado.\n")

    except KeyboardInterrupt:
        print("\nTeste interrompido pelo usuário")
        ars.stop()


if __name__ == "__main__":
    main()
