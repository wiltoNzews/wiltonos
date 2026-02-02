#!/usr/bin/env python3
"""
Daemon de Insights LLM para WiltonOS

Este daemon integra insights baseados em LLM ao sistema WiltonOS,
analisando métricas de phi em tempo real e sugerindo ações corretivas.

Opera em conjunto com o daemon de coerência, fornecendo recomendações
para manter o equilíbrio 3:1 entre coerência e exploração.
"""

import os
import sys
import time
import json
import asyncio
import logging
import argparse
from typing import Dict, Any, Optional

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importar componentes WiltonOS
from wilton_core.llm.llm_insight_hook import LLMInsightHook
from wilton_core.interfaces.hpc_ws_client_auth import HPCWebSocketClientAuth

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("llm_insight_daemon.log")
    ]
)
logger = logging.getLogger("llm_insight_daemon")


class LLMInsightDaemon:
    """
    Daemon que integra insights de LLM ao WiltonOS,
    analisando métricas e sugerindo ações corretivas.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6789,
        use_ssl: bool = False,
        username: str = "admin",
        password: str = "wilton",
        insight_interval: int = 60,  # 1 minuto entre insights
        history_window: int = 10,    # últimas 10 métricas
        auto_refresh: bool = True    # renovação automática de token
    ):
        """
        Inicializa o daemon de insights LLM.
        
        Args:
            host: Endereço do servidor WebSocket
            port: Porta do servidor WebSocket
            use_ssl: Usar SSL para conexão
            username: Nome de usuário para autenticação
            password: Senha para autenticação
            insight_interval: Intervalo entre insights em segundos
            history_window: Quantidade de métricas a analisar
            auto_refresh: Ativar renovação automática de token JWT
        """
        # Criar o hook de insights
        self.insight_hook = LLMInsightHook(
            host=host,
            port=port,
            use_ssl=use_ssl,
            username=username,
            password=password,
            insight_interval=insight_interval,
            history_window=history_window,
            auto_refresh=auto_refresh
        )
        
        # Estado interno
        self.running = False
        logger.info("LLM Insight Daemon inicializado")
    
    async def start(self):
        """Inicia o daemon de insights"""
        if self.running:
            logger.warning("Daemon já está em execução")
            return
            
        self.running = True
        logger.info("Daemon de insights LLM iniciado")
        
        # Iniciar o hook de insights
        await self.insight_hook.run()
    
    async def stop(self):
        """Para o daemon de insights"""
        if not self.running:
            return
            
        self.running = False
        await self.insight_hook.stop()
        logger.info("Daemon de insights LLM parado")


async def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="LLM Insight Daemon para WiltonOS")
    parser.add_argument("--host", default="localhost", help="Endereço do servidor WebSocket")
    parser.add_argument("--port", type=int, default=6789, help="Porta do servidor WebSocket")
    parser.add_argument("--ssl", action="store_true", help="Usar SSL para conexão")
    parser.add_argument("--username", default="admin", help="Nome de usuário para autenticação")
    parser.add_argument("--password", default="wilton", help="Senha para autenticação")
    parser.add_argument("--interval", type=int, default=60, help="Intervalo de geração de insights (segundos)")
    parser.add_argument("--history", type=int, default=10, help="Quantidade de métricas históricas a analisar")
    args = parser.parse_args()
    
    # Criar e iniciar o daemon
    daemon = LLMInsightDaemon(
        host=args.host,
        port=args.port,
        use_ssl=args.ssl,
        username=args.username,
        password=args.password,
        insight_interval=args.interval,
        history_window=args.history
    )
    
    try:
        # Registrar manipulador de interrupção
        loop = asyncio.get_event_loop()
        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
        for s in signals:
            loop.add_signal_handler(s, lambda s=s: asyncio.create_task(daemon.stop()))
        
        logger.info("Pressione Ctrl+C para parar o daemon")
        await daemon.start()
    except KeyboardInterrupt:
        logger.info("Processo interrompido pelo usuário")
    finally:
        await daemon.stop()


if __name__ == "__main__":
    import signal
    asyncio.run(main())