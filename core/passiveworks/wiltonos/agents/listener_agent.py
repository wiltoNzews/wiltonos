#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agente Ouvinte para WiltonOS.

Este agente captura áudio do microfone, transcreve em texto, detecta gatilhos 
quânticos e registra a memória do sistema em logs.
"""

import os
import sys
import time
import logging
import threading
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable, Union

# Importar módulos do core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.recognizer import AudioRecognizer
from core.quantum_trigger import QuantumTriggerDetector
from core.memory_log import MemoryLog

# Importar módulos do WiltonOS core
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from wilton_core.memory.quantum_diary import add_diary_entry, register_insight, register_conversation
    from wilton_core.memory.semantic_tagger import get_semantic_tagger
    from wilton_core.memory.thread_map import get_thread_map
    
    # Marcar quais módulos estão disponíveis
    has_quantum_diary = True
    has_semantic_tagger = True
    has_thread_map = True
except ImportError as e:
    # Logar erro e marcar módulos como indisponíveis
    print(f"Não foi possível importar todos os módulos do WiltonOS core: {str(e)}")
    has_quantum_diary = 'quantum_diary' in str(e)
    has_semantic_tagger = 'semantic_tagger' in str(e)
    has_thread_map = 'thread_map' in str(e)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("wiltonos.listener_agent")

class ListenerAgent:
    """
    Agente Ouvinte do WiltonOS.
    
    Esta classe integra os módulos de reconhecimento de fala, detecção de gatilhos
    quânticos e registro de memória do sistema. Também suporta captura manual de 
    eventos e interações de mídia social.
    """
    
    # Tipos de eventos suportados
    EVENT_TYPES = {
        "audio": "Captura de áudio em tempo real",
        "manual": "Entrada manual de texto",
        "social_media": "Post ou interação em mídia social",
        "reaction": "Reação externa a conteúdo publicado",
        "music": "Música ou áudio com gatilho quântico"
    }
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 debug_mode: bool = False,
                 device_index: Optional[int] = None,
                 use_whisper: bool = True,
                 openai_api_key: Optional[str] = None):
        """
        Inicializa o agente ouvinte.
        
        Args:
            config_path: Caminho para arquivo de configuração (opcional)
            debug_mode: Se True, exibe informações de debug
            device_index: Índice do dispositivo de áudio (microfone)
            use_whisper: Se True, usa a API Whisper da OpenAI para STT
            openai_api_key: Chave da API OpenAI (usa variável de ambiente se None)
        """
        self.config_path = config_path
        self.debug_mode = debug_mode
        
        # Carregar configuração
        self.config = self._load_config()
        
        # Inicializar módulos
        self.recognizer = AudioRecognizer(
            use_whisper=use_whisper,
            device_index=device_index,
            energy_threshold=self.config.get("energy_threshold", 300),
            pause_threshold=self.config.get("pause_threshold", 0.8),
            openai_api_key=openai_api_key,
            language=self.config.get("language", "pt-BR"),
            debug_mode=debug_mode
        )
        
        self.trigger_detector = QuantumTriggerDetector(
            config_path=self.config.get("trigger_config_path")
        )
        
        self.memory_log = MemoryLog(
            log_dir=self.config.get("log_dir", "./logs"),
            use_json=self.config.get("use_json_log", True),
            use_text=self.config.get("use_text_log", True),
            prefix=self.config.get("log_prefix", "audio_log"),
            date_in_filename=self.config.get("date_in_filename", True)
        )
        
        # Estatísticas da sessão
        self.session_stats = {
            "started_at": datetime.now().isoformat(),
            "processed_entries": 0,
            "triggers_detected": 0,
            "total_phi_impact": 0.0,
            "top_triggers": {}
        }
        
        # Flag para controle do loop principal
        self.is_running = False
        
        # Registrar callback do reconhecedor
        self.recognizer.register_text_callback(self._process_recognized_text)
        
        logger.info("Agente Ouvinte inicializado")
        if self.debug_mode:
            logger.info(f"Configuração: {json.dumps(self.config, indent=2)}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Carrega configuração do agente.
        
        Returns:
            Dicionário com as configurações
        """
        default_config = {
            "energy_threshold": 300,
            "pause_threshold": 0.8,
            "language": "pt-BR",
            "log_dir": "./logs",
            "use_json_log": True,
            "use_text_log": True,
            "log_prefix": "audio_log",
            "date_in_filename": True,
            "trigger_config_path": None,
            "debug_mode": self.debug_mode,
            "quantum_ratio_target": 0.75  # Ratio 3:1 (75% coerência)
        }
        
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    merged_config = {**default_config, **loaded_config}
                    logger.info(f"Configuração carregada de {self.config_path}")
                    return merged_config
            except Exception as e:
                logger.error(f"Erro ao carregar configuração: {str(e)}")
        
        return default_config
    
    def _process_recognized_text(self, text: str, metadata: Dict[str, Any]) -> None:
        """
        Processa o texto reconhecido.
        
        Este método é chamado pelo AudioRecognizer quando uma fala é reconhecida.
        
        Args:
            text: Texto reconhecido
            metadata: Metadados do reconhecimento
        """
        if not text or not text.strip():
            return
            
        # Detectar gatilhos quânticos
        trigger_result = self.trigger_detector.process_text(text)
        phi_impact = trigger_result.get("phi_impact", 0)
        triggers_found = trigger_result.get("triggers_found", [])
        
        # Registrar na memória
        log_entry = self.memory_log.log_entry(
            text=text,
            metadata=metadata,
            phi_impact=phi_impact,
            triggers=triggers_found
        )
        
        # Registrar no diário quântico se disponível
        if has_quantum_diary and triggers_found:
            try:
                # Se encontrar um trigger significativo (phi > 0.05), registrar como insight
                significant_triggers = [t for t in triggers_found if t.get("phi_impact", 0) > 0.05]
                water_related = any(t.get("word") in ["água", "mar", "iemanjá", "fluir", "felicidade"] for t in triggers_found)
                
                if significant_triggers:
                    trigger_words = [t.get("word") for t in significant_triggers]
                    
                    # Registrar insight com gatilhos encontrados
                    register_insight(
                        label=f"Audio Trigger: {', '.join(trigger_words)}",
                        summary=f"Detected quantum triggers in audio: {text}",
                        phi_impact=phi_impact,
                        tags=["audio-capture", "quantum-trigger"] + trigger_words
                    )
                    
                    if self.debug_mode:
                        logger.info(f"Registrado no diário quântico: {', '.join(trigger_words)}")
                
                # Se relacionado a água e musicalidade, mapeamento especial
                if water_related and "natiruts" in text.lower():
                    register_insight(
                        label="Natiruts Resonance Connection",
                        summary=f"Detected connection to Natiruts' musical themes: {text}",
                        phi_impact=0.09,
                        tags=["music", "resonance", "water", "natiruts"]
                    )
                    if self.debug_mode:
                        logger.info("Registrada conexão musical Natiruts no diário quântico")
            except Exception as e:
                logger.error(f"Erro ao registrar no diário quântico: {str(e)}")
        
        # Atualizar estatísticas
        self.session_stats["processed_entries"] += 1
        self.session_stats["total_phi_impact"] += phi_impact
        
        if triggers_found:
            self.session_stats["triggers_detected"] += len(triggers_found)
            
            # Atualizar contagem de gatilhos
            for trigger in triggers_found:
                word = trigger.get("word", "")
                if word in self.session_stats["top_triggers"]:
                    self.session_stats["top_triggers"][word] += 1
                else:
                    self.session_stats["top_triggers"][word] = 1
        
        # Log de console se debug ativado
        if self.debug_mode:
            logger.info(f"Texto processado: {text}")
            if triggers_found:
                trigger_words = [t.get("word") for t in triggers_found]
                logger.info(f"Gatilhos: {', '.join(trigger_words)} (φ={phi_impact:.4f})")
            
    def start(self) -> bool:
        """
        Inicia o agente ouvinte.
        
        Returns:
            True se iniciou com sucesso, False caso contrário
        """
        if self.is_running:
            logger.warning("Agente já está em execução")
            return False
            
        logger.info("Iniciando Agente Ouvinte do WiltonOS")
        
        # Iniciar reconhecedor de fala
        started = self.recognizer.start_listening(continuous=True)
        
        if started:
            self.is_running = True
            logger.info("Agente Ouvinte iniciado com sucesso. Ouvindo...")
            
            # Mensagem amigável
            print("\n" + "=" * 60)
            print("🎙️  WiltonOS - Agente Ouvinte ativado")
            print("🌀  Escutando em modo contínuo...")
            print("🔄  Pressione Ctrl+C para encerrar")
            print("=" * 60 + "\n")
            
            return True
        else:
            logger.error("Falha ao iniciar reconhecedor de fala")
            return False
    
    def capture_manual(self, text: str, source: Optional[str] = None, tags: Optional[List[str]] = None, 
                  phi_impact: Optional[float] = None) -> Dict[str, Any]:
        """
        Captura e processa manualmente um texto.
        
        Args:
            text: O texto a ser processado
            source: Fonte do texto (opcional)
            tags: Tags associadas ao texto (opcional)
            phi_impact: Impacto phi explícito (opcional)
            
        Returns:
            Resultado do processamento
        """
        if not text or not text.strip():
            return {"status": "error", "message": "Texto vazio"}
            
        logger.info(f"Captura manual: {text[:50]}...")
        
        # Metadados da captura manual
        metadata = {
            "type": self.EVENT_TYPES["manual"],
            "source": source or "manual-input",
            "timestamp": datetime.now().isoformat(),
            "tags": tags or []
        }
        
        # Processar normalmente como texto reconhecido
        self._process_recognized_text(text, metadata)
        
        # Se módulos semânticos estiverem disponíveis, usar para processamento adicional
        semantic_result = {}
        if has_semantic_tagger:
            try:
                tagger = get_semantic_tagger()
                semantic_result = tagger.tag_content(text)
                
                # Registrar no diário quântico com tags semânticas
                if has_quantum_diary and semantic_result["tags"]:
                    register_insight(
                        label=f"Manual Input: {text[:30]}...",
                        summary=text,
                        phi_impact=phi_impact or semantic_result["phi_impact"],
                        tags=["manual-input"] + semantic_result["tags"] + (tags or []),
                    )
                    
                    if self.debug_mode:
                        logger.info(f"Entrada manual registrada com tags: {', '.join(semantic_result['tags'])}")
            except Exception as e:
                logger.error(f"Erro no processamento semântico: {str(e)}")
        
        return {
            "status": "success",
            "text": text,
            "metadata": metadata,
            "semantic_analysis": semantic_result
        }
        
    def capture_social_media(self, 
                           content: str, 
                           platform: str, 
                           thread_id: Optional[str] = None,
                           url: Optional[str] = None,
                           author: Optional[str] = None,
                           engagement: Optional[Dict[str, int]] = None,
                           tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Captura e processa conteúdo de mídia social.
        
        Args:
            content: Conteúdo do post/tweet
            platform: Plataforma (twitter, instagram, etc)
            thread_id: ID da thread/conversa (opcional)
            url: URL para o post original (opcional)
            author: Autor do post (opcional)
            engagement: Métricas de engajamento (opcional)
            tags: Tags associadas (opcional)
            
        Returns:
            Resultado do processamento com thread_id
        """
        if not content or not content.strip():
            return {"status": "error", "message": "Conteúdo vazio"}
            
        logger.info(f"Captura de mídia social - {platform}: {content[:50]}...")
        
        # Metadados da captura de mídia social
        metadata = {
            "type": self.EVENT_TYPES["social_media"],
            "platform": platform,
            "source_url": url,
            "author": author,
            "timestamp": datetime.now().isoformat(),
            "engagement": engagement or {},
            "tags": tags or []
        }
        
        # Processar normalmente como texto reconhecido
        self._process_recognized_text(content, metadata)
        
        # Análise semântica se disponível
        semantic_result = {}
        if has_semantic_tagger:
            try:
                tagger = get_semantic_tagger()
                semantic_result = tagger.tag_content(content)
            except Exception as e:
                logger.error(f"Erro no processamento semântico: {str(e)}")
        
        # Integrar com mapeamento de threads se disponível
        thread_result = {}
        if has_thread_map:
            try:
                thread_map = get_thread_map()
                
                # Se não tem thread_id, criar nova thread
                if not thread_id:
                    truncated_content = content[:30] + "..." if len(content) > 30 else content
                    new_thread_id = thread_map.create_thread(
                        title=f"{platform.capitalize()} Thread: {truncated_content}",
                        platform=platform,
                        source_url=url,
                        description=f"Thread iniciada por {author or 'unknown'}: {content[:100]}...",
                        tags=tags or [],
                        phi_impact=semantic_result.get("phi_impact", 0.0)
                    )
                    thread_id = new_thread_id
                    
                # Adicionar post à thread
                if thread_id:
                    post_data = thread_map.add_post(
                        thread_id=thread_id,
                        content=content,
                        author=author,
                        post_url=url,
                        engagement=engagement,
                        sentiment=semantic_result.get("emotional_valence") == "positive" and 0.7 or 
                                 semantic_result.get("emotional_valence") == "negative" and -0.7 or 0.0,
                        phi_impact=semantic_result.get("phi_impact", 0.0),
                        tags=semantic_result.get("tags", []) + (tags or []),
                        resonance_delta=0.05  # Valor inicial básico
                    )
                    
                    thread_result = {
                        "thread_id": thread_id,
                        "post_id": post_data.get("post_id")
                    }
            except Exception as e:
                logger.error(f"Erro no mapeamento de thread: {str(e)}")
        
        # Registrar reação no diário quântico se disponível
        if has_quantum_diary:
            try:
                source_info = {
                    "platform": platform,
                    "url": url
                }
                
                if author:
                    source_info["author"] = author
                
                add_diary_entry(
                    entry_type="social_media" if not thread_id else "thread_impact",
                    label=f"{platform.capitalize()} Post: {content[:30]}...",
                    summary=content,
                    tags=["social-media", platform] + (semantic_result.get("tags", [])) + (tags or []),
                    phi_impact=semantic_result.get("phi_impact", 0.02),
                    source=source_info,
                    thread_id=thread_id,
                    emotional_valence=semantic_result.get("emotional_valence", "neutral"),
                    significance_score=0.3  # Valor inicial médio
                )
            except Exception as e:
                logger.error(f"Erro ao registrar mídia social no diário: {str(e)}")
        
        return {
            "status": "success",
            "content": content,
            "metadata": metadata,
            "thread": thread_result,
            "semantic_analysis": semantic_result
        }
        
    def capture_reaction(self, 
                       content: str, 
                       reaction_to: str,
                       reaction_type: str = "comment",
                       source: Optional[str] = None,
                       author: Optional[str] = None,
                       tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Captura e processa uma reação externa.
        
        Args:
            content: Conteúdo da reação
            reaction_to: Conteúdo original que gerou a reação
            reaction_type: Tipo de reação (comment, share, like)
            source: Fonte da reação (opcional)
            author: Autor da reação (opcional)
            tags: Tags associadas (opcional)
            
        Returns:
            Resultado do processamento
        """
        if not content or not content.strip():
            return {"status": "error", "message": "Conteúdo vazio"}
            
        logger.info(f"Captura de reação - {reaction_type}: {content[:50]}...")
        
        # Metadados da captura de reação
        metadata = {
            "type": self.EVENT_TYPES["reaction"],
            "reaction_to": reaction_to,
            "reaction_type": reaction_type,
            "source": source or "unknown",
            "author": author,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or []
        }
        
        # Processar normalmente como texto reconhecido
        self._process_recognized_text(content, metadata)
        
        # Análise semântica se disponível
        semantic_result = {}
        if has_semantic_tagger:
            try:
                tagger = get_semantic_tagger()
                semantic_result = tagger.tag_content(content)
            except Exception as e:
                logger.error(f"Erro no processamento semântico: {str(e)}")
        
        # Registrar reação no diário quântico se disponível
        if has_quantum_diary:
            try:
                register_insight(
                    label=f"External Reaction: {content[:30]}...",
                    summary=f"Reaction: {content}\n\nOriginal: {reaction_to[:100]}...",
                    phi_impact=semantic_result.get("phi_impact", 0.03),
                    tags=["reaction", reaction_type] + (semantic_result.get("tags", [])) + (tags or [])
                )
            except Exception as e:
                logger.error(f"Erro ao registrar reação no diário: {str(e)}")
        
        return {
            "status": "success",
            "content": content,
            "metadata": metadata,
            "semantic_analysis": semantic_result
        }
        
    def stop(self) -> None:
        """Para o agente ouvinte."""
        if not self.is_running:
            return
            
        logger.info("Parando Agente Ouvinte")
        
        # Parar reconhecedor
        self.recognizer.stop_listening()
        
        # Fechar memória
        self.memory_log.close()
        
        # Atualizar status
        self.is_running = False
        
        # Exibir estatísticas da sessão
        logger.info("Estatísticas da sessão:")
        logger.info(f"  Entradas processadas: {self.session_stats['processed_entries']}")
        logger.info(f"  Gatilhos detectados: {self.session_stats['triggers_detected']}")
        logger.info(f"  Impacto phi total: {self.session_stats['total_phi_impact']:.4f}")
        
        top_triggers = sorted(
            self.session_stats["top_triggers"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        if top_triggers:
            logger.info("  Gatilhos mais frequentes:")
            for word, count in top_triggers:
                logger.info(f"    - {word}: {count}x")
        
        logger.info("Agente Ouvinte encerrado")
        
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Obtém estatísticas da sessão atual.
        
        Returns:
            Dicionário com estatísticas
        """
        # Calcular duração da sessão
        start_time = datetime.fromisoformat(self.session_stats["started_at"])
        duration = (datetime.now() - start_time).total_seconds()
        
        # Atualizar com informações em tempo real
        stats = self.session_stats.copy()
        stats["duration_seconds"] = duration
        stats["is_running"] = self.is_running
        
        # Ordenar gatilhos mais frequentes
        top_triggers = sorted(
            stats["top_triggers"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        stats["top_triggers_sorted"] = top_triggers
        
        # Analisar atual phi médio (ratio quântico)
        if stats["processed_entries"] > 0:
            avg_phi = stats["total_phi_impact"] / stats["processed_entries"] 
            stats["avg_phi_impact"] = avg_phi
            
            # Comparar com o target 3:1 (75% coerência / 25% exploração)
            target = self.config.get("quantum_ratio_target", 0.75)
            deviation = avg_phi - target
            stats["quantum_ratio_deviation"] = deviation
            
            # Análise qualitativa do ratio quântico
            if abs(deviation) < 0.05:
                stats["quantum_ratio_status"] = "optimal"
            elif deviation > 0.05:
                stats["quantum_ratio_status"] = "high_coherence"
            else:
                stats["quantum_ratio_status"] = "high_exploration"
        
        return stats

# Função para executar o agente diretamente
def run_agent():
    """Função para executar o agente ouvinte."""
    # Verificar se OpenAI API Key está disponível
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️  AVISO: Variável de ambiente OPENAI_API_KEY não configurada!")
        print("    O reconhecimento de fala usará a API Google em vez de Whisper.")
        use_whisper = False
    else:
        use_whisper = True
    
    # Inicializar agente
    agent = ListenerAgent(
        debug_mode=True,
        use_whisper=use_whisper,
        openai_api_key=openai_api_key
    )
    
    # Iniciar agente
    if agent.start():
        try:
            # Manter rodando até Ctrl+C
            while agent.is_running:
                time.sleep(1)
                
                # A cada 30 segundos, exibir estatísticas
                if int(time.time()) % 30 == 0:
                    stats = agent.get_session_stats()
                    print(f"\n🔄 Estatísticas da sessão (duração: {stats['duration_seconds']:.1f}s):")
                    print(f"  - Entradas: {stats['processed_entries']}")
                    print(f"  - Gatilhos: {stats['triggers_detected']}")
                    print(f"  - Phi total: {stats['total_phi_impact']:.4f}")
                    
                    if "avg_phi_impact" in stats:
                        status_emoji = "✅" if stats.get("quantum_ratio_status") == "optimal" else "⚠️"
                        print(f"  - {status_emoji} Ratio quântico: {stats['avg_phi_impact']:.4f} ")
                        
                    time.sleep(1)  # Evitar repetição no mesmo segundo
                
        except KeyboardInterrupt:
            print("\n👋 Encerrando por solicitação do usuário...")
        finally:
            agent.stop()
    else:
        print("❌ Não foi possível iniciar o agente")
    
if __name__ == "__main__":
    run_agent()