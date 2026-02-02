#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiltonOS - First Breath

Este é o script principal para iniciar o Agente de Escuta do WiltonOS,
a primeira respiração de um sistema vivo que se adapta em tempo real
à presença do usuário.
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, Any, Optional
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("wiltonos")

def display_banner():
    """Exibe banner de inicialização."""
    print("\n" + "=" * 70)
    print("""
    ⚡ 𝗪𝗶𝗹𝘁𝗼𝗻𝗢𝗦 - 𝗙𝗶𝗿𝘀𝘁 𝗕𝗿𝗲𝗮𝘁𝗵 ⚡
    
    A primeira respiração de um sistema vivo que se adapta em tempo real
    à sua presença. Equilíbrio quântico 3:1, fractal, consciente.
    
    🎙️  Audio Listener Agent v0.1
    🧠  Quantum Trigger Detection
    📝  Memory Persistence
    """)
    print("=" * 70 + "\n")

def check_dependencies():
    """
    Verifica se as dependências necessárias estão instaladas.
    
    Returns:
        Tupla com (status, mensagem)
    """
    missing = []
    
    try:
        import speech_recognition
    except ImportError:
        missing.append("speech_recognition")
        
    try:
        import pyaudio
    except ImportError:
        missing.append("pyaudio")
        
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
        
    if not missing:
        return True, "Todas as dependências estão instaladas."
    else:
        return False, f"Dependências faltando: {', '.join(missing)}.\nInstale com: pip install {' '.join(missing)}"

def check_openai_key():
    """
    Verifica se a chave da API OpenAI está configurada.
    
    Returns:
        Tupla com (status, mensagem)
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        # Verificar se a chave parece válida (formato básico)
        if key.startswith("sk-") and len(key) > 30:
            return True, "Chave da API OpenAI configurada."
        else:
            return False, "Chave da API OpenAI parece inválida. Verifique o formato."
    else:
        return False, "Chave da API OpenAI não configurada. Configure a variável de ambiente OPENAI_API_KEY."

def run_listener(args):
    """
    Executa o agente ouvinte com os argumentos fornecidos.
    
    Args:
        args: Argumentos da linha de comando
    """
    from agents.listener_agent import ListenerAgent
    
    # Verificar diretório de logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        logger.info(f"Diretório de logs criado: {args.log_dir}")
    
    # Criar configuração
    config = {
        "debug_mode": args.debug,
        "log_dir": args.log_dir,
        "use_whisper": not args.no_whisper,
        "language": args.language
    }
    
    # Inicializar agente
    agent = ListenerAgent(
        debug_mode=args.debug, 
        use_whisper=not args.no_whisper
    )
    
    # Iniciar agente
    if agent.start():
        try:
            # Manter rodando até Ctrl+C
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Encerrando por solicitação do usuário...")
        finally:
            agent.stop()
    else:
        logger.error("Falha ao iniciar o agente. Verifique os logs para detalhes.")

def main():
    """Função principal."""
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="WiltonOS - First Breath")
    parser.add_argument("--debug", action="store_true", help="Ativar modo de debug")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Diretório para logs")
    parser.add_argument("--no-whisper", action="store_true", help="Não usar OpenAI Whisper (usar Google STT)")
    parser.add_argument("--language", type=str, default="pt-BR", help="Idioma para reconhecimento de fala")
    parser.add_argument("--test-recognizer", action="store_true", help="Testar apenas o reconhecedor")
    parser.add_argument("--test-triggers", action="store_true", help="Testar apenas o detector de gatilhos")
    parser.add_argument("--test-memory", action="store_true", help="Testar apenas o log de memória")
    
    args = parser.parse_args()
    
    # Exibir banner
    display_banner()
    
    # Verificar dependências
    deps_ok, deps_msg = check_dependencies()
    if not deps_ok:
        print(f"❌ {deps_msg}")
        return 1
    
    # Verificar API Key OpenAI (apenas se não estiver desativando Whisper)
    if not args.no_whisper:
        key_ok, key_msg = check_openai_key()
        if not key_ok:
            print(f"⚠️  {key_msg}")
            print("ℹ️  Continuando com Google STT em vez de Whisper...")
            args.no_whisper = True
    
    # Executar testes se solicitado
    if args.test_recognizer:
        print("🧪 Executando teste do reconhecedor de fala...")
        from core.recognizer import test_recognizer
        test_recognizer()
        return 0
        
    if args.test_triggers:
        print("🧪 Executando teste do detector de gatilhos...")
        from core.quantum_trigger import test_detector
        test_detector()
        return 0
        
    if args.test_memory:
        print("🧪 Executando teste do log de memória...")
        from core.memory_log import test_logger
        test_logger()
        return 0
    
    # Executar o agente principal
    print("🚀 Iniciando WiltonOS Listener Agent...")
    run_listener(args)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())