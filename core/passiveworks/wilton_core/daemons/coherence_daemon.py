#!/usr/bin/env python3
"""
Daemon de Auto-Coerência para WiltonOS

Este daemon monitora e mantém a coerência quântica (φ) do sistema,
agendando tarefas automaticamente para manter o equilíbrio 3:1 (75% coerência, 25% exploração).

Funciona como um "ciclo respiratório" natural para o sistema, alternando entre
períodos de coerência e exploração.
"""

import os
import sys
import time
import random
import logging
import asyncio
import argparse
import datetime
from typing import Dict, Any, List, Optional, Tuple

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from wilton_core.interfaces.hpc_ws_client import HPCWebSocketClient

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('coherence_daemon.log')
    ]
)
logger = logging.getLogger('coherence_daemon')


class CoherenceDaemon:
    """
    Daemon que monitora e mantém a coerência quântica (φ) do sistema
    dentro de intervalos específicos.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 6789):
        """
        Inicializa o daemon de coerência.
        
        Args:
            host: Endereço do servidor WebSocket
            port: Porta do servidor WebSocket
        """
        self.host = host
        self.port = port
        self.client = HPCWebSocketClient(host=host, port=port)
        self.connected = False
        
        # Configurações de coerência
        self.target_phi = 0.75  # Alvo principal (75% coerência, 25% exploração)
        self.phi_margin = 0.1   # Margem de tolerância
        self.phi_history = []   # Histórico de leituras de φ
        
        # Estados de alerta
        self.alert_levels = {
            'critical_low': 0.2,   # Abaixo disso é estado crítico
            'low': 0.3,            # Abaixo disso está baixo
            'high': 0.85,          # Acima disso está alto
            'critical_high': 0.9   # Acima disso é estado crítico
        }
        
        # Contadores e métricas
        self.stats = {
            'total_tasks': 0,
            'coherence_boosts': 0,
            'exploration_runs': 0,
            'rebalances': 0,
            'alerts': 0,
            'cycles': 0
        }
        
        # Estado interno
        self.running = False
        self.last_action_time = 0
        self.action_cooldown = 10  # Segundos de espera entre ações
        self.cycle_length = 180    # 3 minutos por ciclo respiratório completo
        self.last_cycle_start = time.time()
        
        # Bandeiras de respiração
        self.inhale_phase = True   # True = fase de inalação (aumento de φ)
        
        logger.info("Daemon de Auto-Coerência inicializado")
    
    async def connect(self) -> bool:
        """
        Conecta ao servidor do HPCManager.
        
        Returns:
            bool: True se conectou com sucesso
        """
        try:
            logger.info(f"Conectando ao HPCManager em {self.host}:{self.port}...")
            self.connected = await self.client.connect()
            
            if self.connected:
                logger.info("✅ Conectado ao HPCManager")
                # Inicializar o HPCManager se necessário
                await self.client.initialize_hpc()
                return True
            else:
                logger.warning("❌ Falha ao conectar ao HPCManager, usando modo de simulação")
                return False
        
        except Exception as e:
            logger.error(f"❌ Erro ao conectar: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """
        Desconecta do servidor.
        """
        if self.connected:
            await self.client.disconnect()
            logger.info("Desconectado do HPCManager")
            self.connected = False
    
    async def get_current_phi(self) -> Dict[str, Any]:
        """
        Obtém o valor atual de φ e informações relacionadas.
        
        Returns:
            Dict: Informações de φ
        """
        try:
            phi_data = await self.client.get_phi(include_history=False)
            current_phi = phi_data.get('current_phi', 0)
            
            # Registrar no histórico
            self.phi_history.append((time.time(), current_phi))
            
            # Manter apenas os últimos 1000 pontos
            if len(self.phi_history) > 1000:
                self.phi_history = self.phi_history[-1000:]
            
            logger.info(f"φ atual: {current_phi:.4f} | Alvo: {self.target_phi:.4f} | Status: {phi_data.get('status', 'unknown')}")
            return phi_data
        
        except Exception as e:
            logger.error(f"Erro ao obter φ: {str(e)}")
            return {'current_phi': 0, 'target_phi': self.target_phi, 'status': 'unknown'}
    
    def _is_in_cooldown(self) -> bool:
        """
        Verifica se há uma ação em cooldown.
        
        Returns:
            bool: True se está em cooldown
        """
        return (time.time() - self.last_action_time) < self.action_cooldown
    
    def _update_cooldown(self) -> None:
        """
        Atualiza o timestamp da última ação.
        """
        self.last_action_time = time.time()
    
    def _calculate_breathing_phase(self) -> Tuple[bool, float]:
        """
        Calcula a fase atual do ciclo respiratório natural.
        O ciclo alterna entre aumentar e diminuir φ para criar uma
        oscilação natural no sistema.
        
        Returns:
            Tuple[bool, float]: (fase de inalação, intensidade)
        """
        # Tempo desde o início do ciclo atual
        elapsed = time.time() - self.last_cycle_start
        
        # Reiniciar ciclo se necessário
        if elapsed > self.cycle_length:
            self.last_cycle_start = time.time()
            elapsed = 0
            self.stats['cycles'] += 1
            logger.info(f"Novo ciclo respiratório iniciado (#{self.stats['cycles']})")
        
        # Calcular posição no ciclo (0 a 1)
        cycle_position = elapsed / self.cycle_length
        
        # Primeira metade é inalação, segunda metade é exalação
        inhale = cycle_position < 0.5
        
        # Intensidade segue uma curva sinusoidal
        # Mais forte no meio de cada fase, mais fraca nas transições
        phase_position = cycle_position % 0.5 / 0.5  # 0 a 1 dentro da fase atual
        intensity = 0.5 - abs(phase_position - 0.5)  # Mais forte no meio (0.5)
        intensity = 0.3 + (intensity * 0.7)  # Escala para 0.3 a 1.0
        
        return (inhale, intensity)
    
    def _get_task_for_breathing(self, current_phi: float) -> Dict[str, Any]:
        """
        Determina a tarefa a ser agendada com base na fase respiratória.
        
        Args:
            current_phi: Valor atual de φ
            
        Returns:
            Dict: Configuração da tarefa
        """
        inhale, intensity = self._calculate_breathing_phase()
        
        # Calcular valores como desvio do alvo
        phi_deviation = abs(current_phi - self.target_phi)
        
        # Verificar se está muito longe do alvo
        is_far_from_target = phi_deviation > self.phi_margin * 2
        
        # Se estiver muito desviado, ignorar fase e corrigir imediatamente
        if is_far_from_target:
            if current_phi < self.target_phi:
                task_type = "coherence_boost"
                priority = 9
                intensity = min(1.0, intensity + phi_deviation)
            else:
                task_type = "exploration_run"
                priority = 9
                intensity = min(1.0, intensity + phi_deviation)
            
            logger.info(f"Desvio significativo detectado (φ={current_phi:.4f}), corrigindo.")
        
        # Seguir fase respiratória normal
        else:
            if inhale:
                # Fase de inalação: aumentar φ com coherence_boost
                task_type = "coherence_boost"
                priority = 6
            else:
                # Fase de exalação: diminuir φ com exploration_run
                task_type = "exploration_run"
                priority = 5
        
        # Construir dados da tarefa
        if task_type == "coherence_boost":
            task_data = {
                "intensity": intensity,
                "duration": int(30 + (intensity * 120)),  # 30-150 segundos
                "target": min(1.0, self.target_phi + 0.05)
            }
        else:  # exploration_run
            task_data = {
                "scope": "adaptive",
                "intensity": intensity,
                "target_domain": random.choice([
                    "fractal_space", "quantum_field", "resonance_network",
                    "coherence_boundary", "integration_zone"
                ])
            }
        
        return {
            "type": task_type,
            "priority": priority,
            "data": task_data
        }
    
    async def _schedule_task(self, task_config: Dict[str, Any]) -> str:
        """
        Agenda uma tarefa no HPCManager.
        
        Args:
            task_config: Configuração da tarefa
            
        Returns:
            str: ID da tarefa
        """
        try:
            task_id = await self.client.schedule_task(
                task_type=task_config['type'],
                task_data=task_config['data'],
                priority=task_config['priority']
            )
            
            self._update_cooldown()
            self.stats['total_tasks'] += 1
            
            if task_config['type'] == 'coherence_boost':
                self.stats['coherence_boosts'] += 1
            elif task_config['type'] == 'exploration_run':
                self.stats['exploration_runs'] += 1
            
            intensity = task_config['data'].get('intensity', 0.5)
            logger.info(f"Tarefa agendada: {task_config['type']} (intensidade: {intensity:.2f}, ID: {task_id})")
            
            return task_id
        
        except Exception as e:
            logger.error(f"Erro ao agendar tarefa: {str(e)}")
            return ""
    
    async def _force_rebalance(self, target_phi: Optional[float] = None) -> bool:
        """
        Força um rebalanceamento do sistema.
        
        Args:
            target_phi: Valor alvo de φ para o rebalanceamento
            
        Returns:
            bool: True se o rebalanceamento foi bem-sucedido
        """
        try:
            result = await self.client.force_rebalance(override_phi=target_phi)
            self._update_cooldown()
            self.stats['rebalances'] += 1
            
            logger.info(f"Rebalanceamento forçado para φ={target_phi or 'padrão'}")
            return True
        
        except Exception as e:
            logger.error(f"Erro ao forçar rebalanceamento: {str(e)}")
            return False
    
    async def check_and_maintain_coherence(self) -> None:
        """
        Verifica e mantém a coerência do sistema.
        """
        # Obter φ atual
        phi_data = await self.get_current_phi()
        current_phi = phi_data.get('current_phi', 0)
        
        # Em cooldown, aguardar
        if self._is_in_cooldown():
            logger.debug("Em período de cooldown, aguardando...")
            return
        
        # Verificar alertas críticos
        if current_phi <= self.alert_levels['critical_low']:
            logger.warning(f"⚠️ ALERTA: φ criticamente baixo ({current_phi:.4f})!")
            self.stats['alerts'] += 1
            
            # Força rebalanceamento imediato
            await self._force_rebalance(target_phi=self.target_phi)
            
            # Agendar tarefas de emergência
            for _ in range(3):
                await self._schedule_task({
                    "type": "coherence_boost",
                    "priority": 10,
                    "data": {
                        "intensity": 1.0,
                        "duration": 180,
                        "target": self.target_phi
                    }
                })
            return
        
        elif current_phi >= self.alert_levels['critical_high']:
            logger.warning(f"⚠️ ALERTA: φ criticamente alto ({current_phi:.4f})!")
            self.stats['alerts'] += 1
            
            # Força rebalanceamento imediato
            await self._force_rebalance(target_phi=self.target_phi)
            
            # Agendar tarefas de emergência
            for _ in range(2):
                await self._schedule_task({
                    "type": "exploration_run",
                    "priority": 10,
                    "data": {
                        "scope": "wide",
                        "intensity": 1.0,
                        "target_domain": "coherence_boundary"
                    }
                })
            return
            
        # Verificar alertas regulares
        elif current_phi <= self.alert_levels['low']:
            logger.info(f"Alerta: φ baixo ({current_phi:.4f})")
            
            # Adicionar tarefa de coerência
            await self._schedule_task({
                "type": "coherence_boost",
                "priority": 8,
                "data": {
                    "intensity": 0.8,
                    "duration": 120,
                    "target": self.target_phi
                }
            })
            return
            
        elif current_phi >= self.alert_levels['high']:
            logger.info(f"Alerta: φ alto ({current_phi:.4f})")
            
            # Adicionar tarefa de exploração
            await self._schedule_task({
                "type": "exploration_run",
                "priority": 7,
                "data": {
                    "scope": "medium",
                    "intensity": 0.7,
                    "target_domain": random.choice([
                        "fractal_space", "quantum_field", "coherence_boundary"
                    ])
                }
            })
            return
        
        # Comportamento normal do ciclo respiratório
        else:
            # Está na faixa normal, seguir ciclo natural
            task = self._get_task_for_breathing(current_phi)
            await self._schedule_task(task)
    
    async def print_status(self) -> None:
        """
        Imprime informações sobre o estado atual do daemon.
        """
        # Obter φ atual
        phi_data = await self.get_current_phi()
        current_phi = phi_data.get('current_phi', 0)
        
        # Calcular fase respiratória
        inhale, intensity = self._calculate_breathing_phase()
        phase_name = "Inalação" if inhale else "Exalação"
        
        # Calcular tempo de execução
        uptime = int(time.time() - self.start_time)
        hours, remainder = divmod(uptime, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        
        # Imprimir status
        logger.info(f"=== Status do Daemon de Auto-Coerência ===")
        logger.info(f"Tempo de execução: {uptime_str}")
        logger.info(f"φ atual: {current_phi:.4f} | Alvo: {self.target_phi:.4f}")
        logger.info(f"Fase respiratória: {phase_name} (intensidade: {intensity:.2f})")
        logger.info(f"Ciclos completos: {self.stats['cycles']}")
        logger.info(f"Tarefas agendadas: {self.stats['total_tasks']} total")
        logger.info(f"  - coherence_boost: {self.stats['coherence_boosts']}")
        logger.info(f"  - exploration_run: {self.stats['exploration_runs']}")
        logger.info(f"Rebalanceamentos: {self.stats['rebalances']}")
        logger.info(f"Alertas: {self.stats['alerts']}")
        logger.info(f"==========================================")
    
    async def run(self, interval: float = 5.0, status_interval: int = 60) -> None:
        """
        Executa o daemon em loop.
        
        Args:
            interval: Intervalo entre verificações em segundos
            status_interval: Intervalo entre impressões de status em segundos
        """
        self.start_time = time.time()
        self.running = True
        last_status_time = 0
        
        try:
            # Conectar ao servidor
            await self.connect()
            
            logger.info(f"Daemon de Auto-Coerência iniciado (intervalo: {interval}s)")
            logger.info(f"Mantendo φ próximo a {self.target_phi:.2f} (75% coerência, 25% exploração)")
            
            # Loop principal
            while self.running:
                # Verificar e manter coerência
                await self.check_and_maintain_coherence()
                
                # Imprimir status periodicamente
                if time.time() - last_status_time > status_interval:
                    await self.print_status()
                    last_status_time = time.time()
                
                # Aguardar próximo ciclo
                await asyncio.sleep(interval)
        
        except KeyboardInterrupt:
            logger.info("Daemon interrompido pelo usuário")
        except Exception as e:
            logger.error(f"Erro no daemon: {str(e)}")
        finally:
            # Desconectar
            self.running = False
            await self.disconnect()
            logger.info("Daemon de Auto-Coerência encerrado")


async def main():
    """
    Função principal para execução do daemon.
    """
    parser = argparse.ArgumentParser(description='Daemon de Auto-Coerência para WiltonOS')
    parser.add_argument('--host', default='localhost', help='Endereço do servidor HPCManager')
    parser.add_argument('--port', type=int, default=6789, help='Porta do servidor HPCManager')
    parser.add_argument('--interval', type=float, default=5.0, help='Intervalo de verificação em segundos')
    parser.add_argument('--status-interval', type=int, default=60, help='Intervalo de status em segundos')
    parser.add_argument('--target-phi', type=float, default=0.75, help='Valor alvo de φ')
    args = parser.parse_args()
    
    # Criar e executar daemon
    daemon = CoherenceDaemon(host=args.host, port=args.port)
    daemon.target_phi = args.target_phi
    
    await daemon.run(interval=args.interval, status_interval=args.status_interval)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDaemon encerrado pelo usuário")