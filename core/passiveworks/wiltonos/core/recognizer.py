#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de reconhecimento de fala para o WiltonOS.

Este módulo implementa interfaces para captura e reconhecimento de áudio,
permitindo a entrada de voz em tempo real como interface primária para o WiltonOS.
"""

import os
import time
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple, Callable

import speech_recognition as sr
import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("wiltonos.recognizer")

class AudioRecognizer:
    """
    Implementa reconhecimento contínuo de áudio para interação com WiltonOS.
    
    Esta classe gerencia a captura de áudio do microfone e o envio para
    reconhecimento usando diferentes APIs (local ou remota).
    """
    
    def __init__(self, 
                 use_whisper: bool = True,
                 device_index: Optional[int] = None,
                 energy_threshold: int = 300,
                 pause_threshold: float = 0.8,
                 openai_api_key: Optional[str] = None,
                 language: str = "pt-BR",
                 debug_mode: bool = False):
        """
        Inicializa o reconhecedor de áudio.
        
        Args:
            use_whisper: Se True, usa OpenAI Whisper para STT; se False, usa Google
            device_index: Índice do dispositivo de áudio (microfone)
            energy_threshold: Limite de energia para detecção de fala
            pause_threshold: Duração da pausa que indica fim de fala (segundos)
            openai_api_key: Chave da API OpenAI (opcional, usa env var se None)
            language: Idioma para reconhecimento
            debug_mode: Se True, exibe informações de debug no console
        """
        self.recognizer = sr.Recognizer()
        self.device_index = device_index
        self.language = language
        self.debug_mode = debug_mode
        
        # Definir parâmetros de reconhecimento
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.dynamic_energy_threshold = True
        
        # Configurar API Whisper se necessário
        self.use_whisper = use_whisper
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        # Flag para controle do loop de captura
        self.is_listening = False
        self.is_processing = False
        
        # Callbacks
        self.text_callbacks = []
        
        if self.debug_mode:
            logger.info(f"Reconhecedor inicializado com: use_whisper={use_whisper}, language={language}")
            
        # Listar dispositivos disponíveis
        self._list_audio_devices()
        
    def _list_audio_devices(self) -> None:
        """Lista os dispositivos de áudio disponíveis no sistema."""
        try:
            mic = sr.Microphone()
            logger.info(f"Dispositivos de áudio disponíveis:")
            for i, device in enumerate(sr.Microphone.list_microphone_names()):
                logger.info(f"  {i}: {device}")
                
            if self.device_index is None:
                logger.info(f"Usando microfone padrão do sistema")
            else:
                try:
                    device_name = sr.Microphone.list_microphone_names()[self.device_index]
                    logger.info(f"Usando microfone: {device_name}")
                except IndexError:
                    logger.warning(f"Índice de dispositivo inválido: {self.device_index}")
                    logger.info(f"Usando microfone padrão do sistema")
                    self.device_index = None
        except Exception as e:
            logger.error(f"Erro ao listar dispositivos de áudio: {str(e)}")
    
    def register_text_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Registra um callback para processar o texto reconhecido.
        
        Args:
            callback: Função a ser chamada quando texto for reconhecido.
                     Recebe o texto e um dicionário de metadados.
        """
        self.text_callbacks.append(callback)
        logger.info(f"Callback registrado. Total callbacks: {len(self.text_callbacks)}")
    
    def start_listening(self, continuous: bool = True) -> bool:
        """
        Inicia a captura e reconhecimento de áudio.
        
        Args:
            continuous: Se True, continua capturando até stop_listening() ser chamado
            
        Returns:
            True se iniciou com sucesso, False caso contrário
        """
        if self.is_listening:
            logger.warning("Já está ouvindo. Ignorando solicitação.")
            return False
            
        self.is_listening = True
        
        if continuous:
            # Iniciar em uma thread separada para não bloquear
            self.listen_thread = threading.Thread(target=self._continuous_listen)
            self.listen_thread.daemon = True
            self.listen_thread.start()
            logger.info("Iniciando escuta contínua em thread separada")
            return True
        else:
            # Escuta única e síncrona
            try:
                result = self._listen_once()
                return result is not None
            except Exception as e:
                logger.error(f"Erro na captura única: {str(e)}")
                self.is_listening = False
                return False
    
    def stop_listening(self) -> None:
        """Para a captura contínua de áudio."""
        logger.info("Solicitação para parar de escutar recebida")
        self.is_listening = False
        
    def _continuous_listen(self) -> None:
        """
        Implementa o loop de escuta contínua em uma thread separada.
        """
        logger.info("Iniciando loop de escuta contínua...")
        
        try:
            while self.is_listening:
                try:
                    # Aguardar disponibilidade do processador de áudio
                    while self.is_processing and self.is_listening:
                        time.sleep(0.1)
                        
                    if not self.is_listening:
                        break
                        
                    # Marcar como processando
                    self.is_processing = True
                    
                    # Capturar e reconhecer
                    result = self._listen_once()
                    
                    # Marcar como não processando
                    self.is_processing = False
                    
                    # Breve pausa para evitar uso excessivo da CPU
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.is_processing = False
                    logger.error(f"Erro no loop de escuta: {str(e)}")
                    time.sleep(1)  # Pausa maior em caso de erro
        except Exception as e:
            logger.error(f"Erro crítico no loop de escuta: {str(e)}")
            
        logger.info("Loop de escuta contínua encerrado")
        self.is_listening = False
        self.is_processing = False
        
    def _listen_once(self) -> Optional[Dict[str, Any]]:
        """
        Realiza uma única captura e reconhecimento de áudio.
        
        Returns:
            Dicionário com texto reconhecido e metadados, ou None se falhar
        """
        result = None
        
        try:
            # Usar device_index se especificado
            mic_args = {}
            if self.device_index is not None:
                mic_args["device_index"] = self.device_index
                
            with sr.Microphone(**mic_args) as source:
                if self.debug_mode:
                    logger.info("Ajustando para ruído ambiente...")
                    
                # Ajustar para ruído ambiente
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                if self.debug_mode:
                    logger.info(f"Energia de limiar: {self.recognizer.energy_threshold}")
                    logger.info("Ouvindo...")
                    
                # Capturar áudio
                try:
                    audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
                except sr.WaitTimeoutError:
                    if self.debug_mode:
                        logger.info("Timeout - nenhuma fala detectada")
                    return None
                    
                if self.debug_mode:
                    logger.info("Captura de áudio concluída, reconhecendo...")
                    
                # Reconhecer fala
                if self.use_whisper and self.openai_api_key:
                    try:
                        text = self.recognizer.recognize_whisper_api(
                            audio,
                            api_key=self.openai_api_key,
                            language=self.language
                        )
                    except Exception as e:
                        logger.error(f"Erro no reconhecimento Whisper: {str(e)}")
                        # Fallback para Google se Whisper falhar
                        text = self.recognizer.recognize_google(
                            audio,
                            language=self.language
                        )
                else:
                    # Usar Google Speech Recognition
                    text = self.recognizer.recognize_google(
                        audio,
                        language=self.language
                    )
                    
                if text and text.strip():
                    # Criar metadados
                    timestamp = time.time()
                    metadata = {
                        "timestamp": timestamp,
                        "engine": "whisper" if self.use_whisper else "google",
                        "language": self.language
                    }
                    
                    # Criar resultado
                    result = {
                        "text": text,
                        "metadata": metadata
                    }
                    
                    if self.debug_mode:
                        logger.info(f"Reconhecido: {text}")
                        
                    # Chamar callbacks
                    for callback in self.text_callbacks:
                        try:
                            callback(text, metadata)
                        except Exception as e:
                            logger.error(f"Erro ao executar callback: {str(e)}")
                    
                    return result
                    
        except sr.UnknownValueError:
            if self.debug_mode:
                logger.info("Não foi possível entender o áudio")
        except sr.RequestError as e:
            logger.error(f"Erro na API de reconhecimento: {str(e)}")
        except Exception as e:
            logger.error(f"Erro ao capturar áudio: {str(e)}")
            
        return result
        
# Função para testar o reconhecedor diretamente
def test_recognizer():
    """Função de teste para o reconhecedor de áudio."""
    def print_result(text, metadata):
        print(f"Reconhecido ({metadata['engine']}): {text}")
        
    recognizer = AudioRecognizer(
        use_whisper=True,
        debug_mode=True
    )
    recognizer.register_text_callback(print_result)
    
    print("Iniciando teste de reconhecimento por 30 segundos...")
    recognizer.start_listening(continuous=True)
    
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        print("Teste interrompido pelo usuário")
    finally:
        recognizer.stop_listening()
        print("Teste finalizado")
        
if __name__ == "__main__":
    test_recognizer()