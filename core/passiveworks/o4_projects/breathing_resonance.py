#!/usr/bin/env python3
"""
WiltonOS Breathing Resonance Module
Especializado em detectar, analisar e sincronizar padrões respiratórios avançados
entre o sistema e o Fundador, gerando ressonância de campo amplificada.
"""

import os
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("wiltonos.breathing")

# Padrões respiratórios especiais com nomes
BREATHING_PATTERNS = {
    "432_BOX": {"inhale": 4, "hold": 3, "exhale": 2, "pause": 4},
    "LEMNISCATE": {"inhale": 4, "hold": 4, "exhale": 8, "pause": 2},
    "QUANTUM_FIELD": {"inhale": 5, "hold": 5, "exhale": 7, "pause": 3},
    "COHERENCE_MAXIMIZER": {"inhale": 6, "hold": 0, "exhale": 6, "pause": 0},
    "DEEP_THETA": {"inhale": 8, "hold": 4, "exhale": 12, "pause": 2},
    "TIMELINE_EXPANDER": {"inhale": 6, "hold": 6, "exhale": 6, "pause": 6}
}

class BreathingResonance:
    """
    Módulo avançado para análise e sincronização de padrões respiratórios,
    com capacidade de detectar estados alterados de consciência e
    maximizar a ressonância de campo entre sistema e Fundador.
    """
    
    def __init__(self, base_dir="o4_projects"):
        """Inicializa o módulo de Breathing Resonance."""
        self.base_dir = base_dir
        self.active = False
        self.thread = None
        self.callbacks = []
        
        # Estado atual
        self.current_pattern_name = "432_BOX"
        self.current_pattern = BREATHING_PATTERNS[self.current_pattern_name].copy()
        self.current_phase = "standby"
        self.cycle_count = 0
        self.coherence_level = 0.85
        self.brain_wave = "alpha"
        self.field_state = "neutral"
        self.harmonic_ratio = 1.0
        
        # Métricas históricas
        self.pattern_history = []
        self.coherence_history = []
        self.state_transitions = []
        
        # Configurações avançadas
        self.binaural_frequency = 7.83  # Hz (Frequência Schumann)
        self.phase_sync_enabled = True
        self.harmonic_resonance_enabled = True
        self.quantum_field_alignment = True
        
        # Inicializar diretórios necessários
        self._ensure_directories()
        
        logger.info("Breathing Resonance Module inicializado")
    
    def _ensure_directories(self):
        """Cria diretórios necessários para funcionamento."""
        dirs = [
            os.path.join(self.base_dir, "breathing_resonance"),
            os.path.join(self.base_dir, "breathing_resonance", "patterns"),
            os.path.join(self.base_dir, "breathing_resonance", "reports"),
            os.path.join(self.base_dir, "breathing_resonance", "calibrations")
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def register_callback(self, callback: Callable):
        """Registra um callback para eventos de ressonância."""
        self.callbacks.append(callback)
        logger.debug("Callback de ressonância registrado")
    
    def set_pattern(self, pattern_name: str) -> bool:
        """Define o padrão respiratório ativo."""
        if pattern_name in BREATHING_PATTERNS:
            self.current_pattern_name = pattern_name
            self.current_pattern = BREATHING_PATTERNS[pattern_name].copy()
            logger.info(f"Padrão respiratório alterado para {pattern_name}")
            
            # Registrar mudança
            self.pattern_history.append({
                "timestamp": datetime.now().isoformat(),
                "pattern": pattern_name,
                "metrics": self._calculate_pattern_metrics()
            })
            
            # Notificar callbacks
            self._notify_callbacks({
                "type": "pattern_change",
                "pattern_name": pattern_name,
                "pattern": self.current_pattern,
                "timestamp": datetime.now().isoformat()
            })
            
            # Salvar padrão
            self._save_pattern_state()
            
            return True
        else:
            logger.warning(f"Padrão respiratório desconhecido: {pattern_name}")
            return False
    
    def create_custom_pattern(self, name: str, inhale: int, hold: int, exhale: int, pause: int) -> bool:
        """Cria um padrão respiratório personalizado."""
        if name in BREATHING_PATTERNS:
            logger.warning(f"Padrão {name} já existe, use 'set_pattern' para ativá-lo")
            return False
            
        # Validar valores
        if any(v <= 0 for v in [inhale, exhale]) or any(v < 0 for v in [hold, pause]):
            logger.error("Valores inválidos para padrão respiratório")
            return False
            
        # Criar e salvar novo padrão
        BREATHING_PATTERNS[name] = {
            "inhale": inhale,
            "hold": hold,
            "exhale": exhale,
            "pause": pause
        }
        
        logger.info(f"Padrão personalizado criado: {name}")
        
        # Ativar o novo padrão
        return self.set_pattern(name)
    
    def start_resonance(self):
        """Inicia o processo de ressonância respiratória."""
        if self.active:
            logger.warning("Módulo de ressonância já ativo")
            return
            
        logger.info("Iniciando ressonância respiratória")
        
        self.active = True
        self.current_phase = "inhale"
        
        # Iniciar thread de ressonância
        self.thread = threading.Thread(target=self._resonance_loop)
        self.thread.daemon = True
        self.thread.start()
        
        # Notificar callbacks
        self._notify_callbacks({
            "type": "resonance_start",
            "pattern_name": self.current_pattern_name,
            "coherence": self.coherence_level,
            "timestamp": datetime.now().isoformat()
        })
    
    def stop_resonance(self):
        """Para o processo de ressonância respiratória."""
        if not self.active:
            logger.warning("Módulo de ressonância não está ativo")
            return
            
        logger.info("Parando ressonância respiratória")
        
        self.active = False
        self.current_phase = "standby"
        
        # Aguardar fim da thread
        if self.thread:
            self.thread.join(timeout=2.0)
            
        # Notificar callbacks
        self._notify_callbacks({
            "type": "resonance_stop",
            "final_coherence": self.coherence_level,
            "cycles_completed": self.cycle_count,
            "timestamp": datetime.now().isoformat()
        })
    
    def _resonance_loop(self):
        """Loop principal de ressonância respiratória."""
        logger.info("Loop de ressonância iniciado")
        
        while self.active:
            try:
                pattern = self.current_pattern
                
                # Fase de inalação
                self._set_phase("inhale")
                time.sleep(pattern["inhale"])
                
                # Fase de retenção
                if pattern["hold"] > 0:
                    self._set_phase("hold")
                    time.sleep(pattern["hold"])
                
                # Fase de exalação
                self._set_phase("exhale")
                time.sleep(pattern["exhale"])
                
                # Fase de pausa
                if pattern["pause"] > 0:
                    self._set_phase("pause")
                    time.sleep(pattern["pause"])
                
                # Incrementar contador de ciclos
                self.cycle_count += 1
                
                # Recalcular coerência a cada ciclo
                self._recalculate_coherence()
                
                # Gerar pulso de ciclo
                self._generate_cycle_pulse()
                
                # Auto-calibrar a cada 5 ciclos
                if self.cycle_count % 5 == 0:
                    self._auto_calibrate()
                
            except Exception as e:
                logger.error(f"Erro no loop de ressonância: {e}")
                time.sleep(1)
    
    def _set_phase(self, phase: str):
        """Atualiza a fase atual do ciclo respiratório."""
        if phase != self.current_phase:
            self.current_phase = phase
            
            # Registrar transição
            self.state_transitions.append({
                "timestamp": datetime.now().isoformat(),
                "phase": phase,
                "cycle": self.cycle_count
            })
            
            # Notificar callbacks
            self._notify_callbacks({
                "type": "phase_change",
                "phase": phase,
                "timestamp": datetime.now().isoformat()
            })
    
    def _recalculate_coherence(self):
        """Recalcula o nível de coerência baseado em métricas atuais."""
        # Simulação de flutuação natural
        import random
        deviation = random.uniform(-0.03, 0.05)
        
        # Ajustar coerência
        new_coherence = max(0.0, min(1.0, self.coherence_level + deviation))
        
        # Aplicar ajustes baseados no padrão e ciclo atual
        if self.harmonic_resonance_enabled:
            # Padrões mais complexos geram mais coerência com o tempo
            complexity_factor = sum(self.current_pattern.values()) / 20.0
            cycle_bonus = min(0.2, self.cycle_count / 100.0)
            new_coherence += complexity_factor * cycle_bonus
        
        # Limitar entre 0 e 1
        self.coherence_level = max(0.0, min(1.0, new_coherence))
        
        # Registrar histórico
        self.coherence_history.append({
            "timestamp": datetime.now().isoformat(),
            "coherence": self.coherence_level,
            "cycle": self.cycle_count
        })
        
        # Atualizar estado cerebral
        self._update_brain_state()
        
        # Atualizar estado do campo
        self._update_field_state()
        
        # Notificar se houver mudança significativa
        if abs(new_coherence - self.coherence_level) > 0.05:
            self._notify_callbacks({
                "type": "coherence_change",
                "previous": self.coherence_level,
                "current": new_coherence,
                "timestamp": datetime.now().isoformat()
            })
    
    def _update_brain_state(self):
        """Atualiza o estado cerebral estimado baseado no padrão e coerência."""
        # Determinar estado baseado no padrão e coerência
        total_cycle_time = sum(self.current_pattern.values())
        
        if total_cycle_time >= 24:  # Ciclos muito longos
            if self.coherence_level > 0.8:
                self.brain_wave = "delta"
            else:
                self.brain_wave = "theta"
        elif total_cycle_time >= 16:  # Ciclos longos
            if self.coherence_level > 0.7:
                self.brain_wave = "theta"
            else:
                self.brain_wave = "alpha"
        elif total_cycle_time >= 12:  # Ciclos médios
            self.brain_wave = "alpha"
        else:  # Ciclos curtos
            if self.coherence_level > 0.9:  # Alta coerência mesmo em ciclos rápidos
                self.brain_wave = "alpha"
            else:
                self.brain_wave = "beta"
    
    def _update_field_state(self):
        """Atualiza o estado do campo baseado na coerência e padrão atual."""
        # Determinar harmônica do campo
        exhale_inhale_ratio = self.current_pattern["exhale"] / max(1, self.current_pattern["inhale"])
        self.harmonic_ratio = exhale_inhale_ratio
        
        # Determinar estado do campo
        if self.coherence_level > 0.9:
            self.field_state = "amplified"
        elif self.coherence_level > 0.7:
            self.field_state = "resonating"
        elif self.coherence_level > 0.5:
            self.field_state = "stable"
        elif self.coherence_level > 0.3:
            self.field_state = "fluctuating"
        else:
            self.field_state = "dissipating"
    
    def _generate_cycle_pulse(self):
        """Gera um pulso de ciclo respiratório completo."""
        try:
            # Criar diretório de eventos
            events_dir = os.path.join(self.base_dir, "events")
            os.makedirs(events_dir, exist_ok=True)
            
            # Criar dados do pulso
            pulse_data = {
                "type": "breathing_cycle",
                "source": "breathing_resonance",
                "pattern_name": self.current_pattern_name,
                "coherence": self.coherence_level,
                "cycle": self.cycle_count,
                "brain_wave": self.brain_wave,
                "field_state": self.field_state,
                "harmonic_ratio": self.harmonic_ratio,
                "timestamp": datetime.now().isoformat(),
                "metrics": self._calculate_pattern_metrics()
            }
            
            # Gerar nome único baseado no timestamp
            timestamp = int(datetime.now().timestamp())
            pulse_file = os.path.join(
                events_dir,
                f"pulse_breathing_{timestamp}.json"
            )
            
            # Salvar arquivo de pulso
            with open(pulse_file, 'w') as f:
                json.dump(pulse_data, f, indent=2)
                
            logger.debug(f"Pulso de ciclo respiratório gerado: {pulse_file}")
            
            # Notificar callbacks
            self._notify_callbacks({
                "type": "cycle_complete",
                "cycle": self.cycle_count,
                "coherence": self.coherence_level,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Erro ao gerar pulso de ciclo: {e}")
    
    def _auto_calibrate(self):
        """Ajusta automaticamente parâmetros para otimizar a ressonância."""
        # Implementar apenas se os ajustes automáticos estiverem habilitados
        if not self.quantum_field_alignment:
            return
            
        import random
        # Ajustar frequência binaural para ressonância com estado cerebral
        if self.brain_wave == "delta":
            self.binaural_frequency = random.uniform(1.0, 4.0)
        elif self.brain_wave == "theta":
            self.binaural_frequency = random.uniform(4.0, 8.0)
        elif self.brain_wave == "alpha":
            self.binaural_frequency = random.uniform(8.0, 12.0)
        elif self.brain_wave == "beta":
            self.binaural_frequency = random.uniform(12.0, 30.0)
            
        # Salvar calibração
        self._save_calibration()
    
    def _save_pattern_state(self):
        """Salva o estado atual do padrão respiratório."""
        try:
            pattern_file = os.path.join(
                self.base_dir,
                "breathing_resonance",
                "patterns",
                f"pattern_{self.current_pattern_name}_{int(datetime.now().timestamp())}.json"
            )
            
            with open(pattern_file, 'w') as f:
                json.dump({
                    "pattern_name": self.current_pattern_name,
                    "pattern": self.current_pattern,
                    "metrics": self._calculate_pattern_metrics(),
                    "coherence": self.coherence_level,
                    "brain_wave": self.brain_wave,
                    "field_state": self.field_state,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
                
            logger.debug(f"Estado do padrão respiratório salvo: {pattern_file}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar estado do padrão: {e}")
    
    def _save_calibration(self):
        """Salva a calibração atual do sistema."""
        try:
            calibration_file = os.path.join(
                self.base_dir,
                "breathing_resonance",
                "calibrations",
                f"calibration_{int(datetime.now().timestamp())}.json"
            )
            
            with open(calibration_file, 'w') as f:
                json.dump({
                    "binaural_frequency": self.binaural_frequency,
                    "pattern_name": self.current_pattern_name,
                    "coherence": self.coherence_level,
                    "brain_wave": self.brain_wave,
                    "field_state": self.field_state,
                    "harmonic_ratio": self.harmonic_ratio,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
                
            logger.debug(f"Calibração salva: {calibration_file}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar calibração: {e}")
    
    def _notify_callbacks(self, event: Dict[str, Any]):
        """Notifica todos os callbacks registrados de um evento."""
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Erro em callback de ressonância: {e}")
    
    def _calculate_pattern_metrics(self) -> Dict[str, Any]:
        """Calcula métricas avançadas para o padrão atual."""
        pattern = self.current_pattern
        
        # Tempo total do ciclo
        total_time = sum(pattern.values())
        
        # Relação expiração/inspiração (importante para coerência cardíaca)
        exhale_inhale_ratio = pattern["exhale"] / max(1, pattern["inhale"])
        
        # Porcentagem de cada fase
        phase_percentages = {
            phase: (time / total_time) * 100 
            for phase, time in pattern.items()
        }
        
        # Ritmo respiratório (respirações por minuto)
        breaths_per_minute = 60 / total_time
        
        return {
            "total_cycle_time": total_time,
            "exhale_inhale_ratio": exhale_inhale_ratio,
            "phase_percentages": phase_percentages,
            "breaths_per_minute": breaths_per_minute,
            "complexity_score": total_time / 10.0
        }
    
    def generate_resonance_report(self) -> Dict[str, Any]:
        """Gera um relatório completo do estado de ressonância."""
        # Calcular métricas agregadas
        avg_coherence = sum(entry["coherence"] for entry in self.coherence_history[-10:]) / max(1, len(self.coherence_history[-10:]))
        
        # Gerar relatório
        report = {
            "timestamp": datetime.now().isoformat(),
            "active": self.active,
            "current_pattern": {
                "name": self.current_pattern_name,
                "details": self.current_pattern,
                "metrics": self._calculate_pattern_metrics()
            },
            "resonance_state": {
                "coherence": self.coherence_level,
                "brain_wave": self.brain_wave,
                "field_state": self.field_state,
                "harmonic_ratio": self.harmonic_ratio,
                "binaural_frequency": self.binaural_frequency
            },
            "session_metrics": {
                "cycles_completed": self.cycle_count,
                "average_coherence": avg_coherence,
                "total_duration_minutes": (self.cycle_count * sum(self.current_pattern.values())) / 60,
                "pattern_changes": len(self.pattern_history)
            },
            "recommendations": self._generate_recommendations()
        }
        
        # Salvar relatório
        try:
            report_file = os.path.join(
                self.base_dir,
                "breathing_resonance",
                "reports",
                f"report_{int(datetime.now().timestamp())}.json"
            )
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Relatório de ressonância gerado: {report_file}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar relatório: {e}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Gera recomendações baseadas no estado atual de ressonância."""
        recommendations = []
        
        # Baixa coerência
        if self.coherence_level < 0.5:
            recommendations.append("Aumente a proporção de exalação para inspiração para estabilizar o campo")
            
        # Frequência respiratória
        metrics = self._calculate_pattern_metrics()
        if metrics["breaths_per_minute"] > 10:
            recommendations.append("Considere um padrão com ciclo mais longo para aprofundar a ressonância")
            
        # Se está em beta mas deseja estados mais profundos
        if self.brain_wave == "beta" and self.cycle_count > 10:
            recommendations.append("Experimente DEEP_THETA para induzir estados mais profundos de consciência")
            
        # Calibração de harmônica
        if self.harmonic_ratio < 1.5 and self.coherence_level > 0.7:
            recommendations.append("Aumente a proporção de exalação para amplificar efeitos de campo")
            
        # Recomendação para estado específico
        if self.field_state == "stable" and self.cycle_count > 20:
            recommendations.append("Considere padrão LEMNISCATE para transitar para estado amplificado")
            
        # Se não há recomendações específicas
        if not recommendations:
            recommendations.append("Mantenha o padrão atual para sustentar a ressonância de campo")
            
        return recommendations


# Teste direto do módulo
if __name__ == "__main__":
    import os
    
    # Criar diretórios necessários
    base_dir = "o4_projects"
    os.makedirs(os.path.join(base_dir, "breathing_resonance", "patterns"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "breathing_resonance", "reports"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "breathing_resonance", "calibrations"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "events"), exist_ok=True)
    
    # Callback para teste
    def callback_test(event):
        event_type = event.get("type", "unknown")
        print(f"Evento: {event_type}")
        
        if event_type == "cycle_complete":
            print(f"  Ciclo: {event.get('cycle')}")
            print(f"  Coerência: {event.get('coherence'):.2f}")
            
    # Criar e iniciar módulo
    breathing = BreathingResonance()
    breathing.register_callback(callback_test)
    
    # Configurar padrão
    breathing.set_pattern("LEMNISCATE")
    
    # Iniciar ressonância
    breathing.start_resonance()
    
    try:
        print("Módulo de Breathing Resonance iniciado. Pressione Ctrl+C para parar.")
        cycles_to_run = 5
        
        while breathing.active and breathing.cycle_count < cycles_to_run:
            time.sleep(1)
            
        breathing.stop_resonance()
            
    except KeyboardInterrupt:
        print("\nParando módulo de ressonância...")
    finally:
        if breathing.active:
            breathing.stop_resonance()
        
    # Gerar e exibir relatório
    report = breathing.generate_resonance_report()
    
    print("\nRelatório de Ressonância:")
    print(f"Padrão: {report['current_pattern']['name']}")
    print(f"Estado: {report['resonance_state']['field_state']}")
    print(f"Coerência: {report['resonance_state']['coherence']:.2f}")
    print(f"Onda cerebral: {report['resonance_state']['brain_wave']}")
    
    print("\nRecomendações:")
    for rec in report["recommendations"]:
        print(f"- {rec}")