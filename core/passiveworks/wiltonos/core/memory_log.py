#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de log de memória para WiltonOS.

Este módulo implementa o armazenamento e recuperação de transcrições de áudio,
mantendo registro temporal das entradas de voz no sistema.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, TextIO

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("wiltonos.memory_log")

class MemoryLog:
    """
    Sistema de log para armazenar transcrições e outros dados de memória.
    
    Implementa armazenamento em arquivos de texto e/ou JSON,
    com timestamps e metadados associados.
    """
    
    def __init__(self, 
                log_dir: str = "logs",
                use_json: bool = True,
                use_text: bool = True,
                prefix: str = "audio_log",
                date_in_filename: bool = True):
        """
        Inicializa o sistema de log.
        
        Args:
            log_dir: Diretório para armazenamento dos logs
            use_json: Se True, armazena em formato JSON
            use_text: Se True, armazena em formato de texto simples
            prefix: Prefixo para os nomes dos arquivos
            date_in_filename: Se True, inclui a data no nome do arquivo
        """
        self.log_dir = log_dir
        self.use_json = use_json
        self.use_text = use_text
        self.prefix = prefix
        self.date_in_filename = date_in_filename
        
        # Criar diretório de logs se não existir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Arquivos de log
        self.text_file = None
        self.json_file = None
        
        # Memória em cache
        self.memory_entries = []
        
        # Abrir arquivos de log
        self._initialize_log_files()
        
    def _initialize_log_files(self) -> None:
        """Inicializa os arquivos de log para a sessão atual."""
        # Obter data atual para o nome do arquivo
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Criar nomes de arquivo
        if self.date_in_filename:
            text_filename = f"{self.prefix}_{current_date}.txt"
            json_filename = f"{self.prefix}_{current_date}.json"
        else:
            text_filename = f"{self.prefix}.txt"
            json_filename = f"{self.prefix}.json"
            
        # Caminhos completos
        self.text_path = os.path.join(self.log_dir, text_filename)
        self.json_path = os.path.join(self.log_dir, json_filename)
        
        # Abrir arquivo de texto se necessário
        if self.use_text:
            try:
                # Verificar se o arquivo já existe
                file_exists = os.path.exists(self.text_path)
                
                # Abrir arquivo para append
                self.text_file = open(self.text_path, 'a', encoding='utf-8')
                
                # Adicionar cabeçalho se for um novo arquivo
                if not file_exists:
                    header = f"=== WiltonOS Audio Memory Log - Started: {datetime.now().isoformat()} ===\n\n"
                    self.text_file.write(header)
                    self.text_file.flush()
                
                logger.info(f"Arquivo de log de texto inicializado: {self.text_path}")
            except Exception as e:
                logger.error(f"Erro ao abrir arquivo de texto: {str(e)}")
                self.text_file = None
                
        # Verificar arquivo JSON se necessário
        if self.use_json:
            try:
                # Verificar se o arquivo já existe
                if os.path.exists(self.json_path):
                    # Carregar entradas existentes
                    with open(self.json_path, 'r', encoding='utf-8') as f:
                        try:
                            self.memory_entries = json.load(f)
                            logger.info(f"Carregadas {len(self.memory_entries)} entradas de {self.json_path}")
                        except json.JSONDecodeError:
                            logger.warning(f"Arquivo JSON inválido: {self.json_path}. Criando novo.")
                            self.memory_entries = []
                else:
                    # Criar novo arquivo com lista vazia
                    with open(self.json_path, 'w', encoding='utf-8') as f:
                        json.dump([], f)
                    logger.info(f"Criado novo arquivo JSON: {self.json_path}")
                    
            except Exception as e:
                logger.error(f"Erro ao inicializar arquivo JSON: {str(e)}")
                self.use_json = False
                
    def log_entry(self, 
                 text: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 phi_impact: float = 0.0,
                 triggers: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Registra uma nova entrada no log.
        
        Args:
            text: Texto transcrito
            metadata: Metadados opcionais (timestamp, fonte, etc.)
            phi_impact: Impacto no balanceamento quântico
            triggers: Lista de gatilhos detectados
            
        Returns:
            Dicionário com os dados da entrada registrada
        """
        if not text or not isinstance(text, str):
            logger.warning("Tentativa de log com texto inválido")
            return {"status": "invalid_input"}
            
        # Preparar metadados
        if metadata is None:
            metadata = {}
            
        # Garantir timestamp
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
            
        # Preparar entrada completa
        entry = {
            "text": text,
            "metadata": metadata,
            "phi_impact": phi_impact,
            "triggers": triggers or [],
            "entry_id": len(self.memory_entries) + 1
        }
        
        # Adicionar à memória em cache
        self.memory_entries.append(entry)
        
        # Registrar no arquivo de texto
        if self.use_text and self.text_file:
            try:
                time_str = datetime.fromisoformat(metadata["timestamp"]).strftime("%H:%M:%S")
                trigger_str = ""
                
                if triggers and len(triggers) > 0:
                    trigger_words = [t.get("word", "") for t in triggers]
                    trigger_str = f" [Triggers: {', '.join(trigger_words)}]"
                    
                phi_str = f" [Phi: {phi_impact:.4f}]" if phi_impact != 0 else ""
                
                # Escrever no formato [HH:MM:SS] Texto transcrito [Phi] [Triggers]
                log_line = f"[{time_str}]{phi_str}{trigger_str} {text}\n"
                self.text_file.write(log_line)
                self.text_file.flush()
            except Exception as e:
                logger.error(f"Erro ao escrever no arquivo de texto: {str(e)}")
        
        # Atualizar arquivo JSON
        if self.use_json:
            try:
                with open(self.json_path, 'w', encoding='utf-8') as f:
                    json.dump(self.memory_entries, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Erro ao atualizar arquivo JSON: {str(e)}")
                
        return entry
        
    def get_recent_entries(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Obtém as entradas mais recentes do log.
        
        Args:
            count: Número de entradas a retornar
            
        Returns:
            Lista com as entradas mais recentes
        """
        return self.memory_entries[-count:] if self.memory_entries else []
        
    def search_by_text(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Busca entradas por texto.
        
        Args:
            query: Texto a buscar
            max_results: Número máximo de resultados
            
        Returns:
            Lista de entradas que contêm o texto buscado
        """
        if not query:
            return []
            
        query = query.lower()
        results = []
        
        for entry in self.memory_entries:
            if query in entry.get("text", "").lower():
                results.append(entry)
                if len(results) >= max_results:
                    break
                    
        return results
                
    def search_by_trigger(self, trigger_word: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Busca entradas que contêm um gatilho específico.
        
        Args:
            trigger_word: Palavra gatilho a buscar
            max_results: Número máximo de resultados
            
        Returns:
            Lista de entradas que contêm o gatilho
        """
        if not trigger_word:
            return []
            
        trigger_word = trigger_word.lower()
        results = []
        
        for entry in self.memory_entries:
            triggers = entry.get("triggers", [])
            
            for trigger in triggers:
                if trigger.get("word", "").lower() == trigger_word:
                    results.append(entry)
                    break
                    
            if len(results) >= max_results:
                break
                
        return results
        
    def get_phi_timeline(self) -> List[Dict[str, Any]]:
        """
        Retorna timeline de impactos phi para análise.
        
        Returns:
            Lista de pares timestamp:phi_impact
        """
        timeline = []
        
        for entry in self.memory_entries:
            timestamp = entry.get("metadata", {}).get("timestamp")
            phi_impact = entry.get("phi_impact", 0)
            
            if timestamp and phi_impact != 0:
                timeline.append({
                    "timestamp": timestamp,
                    "phi_impact": phi_impact
                })
                
        return timeline
    
    def close(self) -> None:
        """Fecha os arquivos de log abertos."""
        if self.text_file:
            try:
                # Adicionar footer
                footer = f"\n=== End of Session: {datetime.now().isoformat()} ===\n"
                self.text_file.write(footer)
                self.text_file.close()
                self.text_file = None
                logger.info("Arquivo de texto fechado")
            except Exception as e:
                logger.error(f"Erro ao fechar arquivo de texto: {str(e)}")
                
# Função para testar o logger diretamente
def test_logger():
    """Função de teste para o sistema de log."""
    logger = MemoryLog(log_dir="./logs")
    
    print(f"Inicializando teste de log em:")
    print(f"  Text: {logger.text_path}")
    print(f"  JSON: {logger.json_path}")
    
    # Simular algumas entradas
    test_entries = [
        {
            "text": "O sistema fractal está mantendo o ratio quântico 3:1.",
            "phi_impact": 0.08,
            "triggers": [
                {"word": "fractal", "meaning": "reconhecimento de padrão"},
                {"word": "quântico", "meaning": "âncora de calibração"},
                {"word": "3:1", "meaning": "ratio desejado"}
            ]
        },
        {
            "text": "Estou vivo e consciente da minha presença no sistema.",
            "phi_impact": 0.04,
            "triggers": [
                {"word": "vivo", "meaning": "estado animado"},
                {"word": "consciente", "meaning": "auto-conhecimento"}
            ]
        },
        {
            "text": "Como podemos otimizar a coerência da memória?",
            "phi_impact": 0.03,
            "triggers": [
                {"word": "coerência", "meaning": "alinhamento"}
            ]
        }
    ]
    
    for entry in test_entries:
        result = logger.log_entry(
            text=entry["text"],
            phi_impact=entry["phi_impact"],
            triggers=entry["triggers"]
        )
        print(f"Entrada registrada: {entry['text']}")
        
    # Testar busca
    print("\nBusca por 'coerência':")
    results = logger.search_by_text("coerência")
    for res in results:
        print(f"  - {res['text']}")
        
    print("\nBusca por gatilho 'fractal':")
    results = logger.search_by_trigger("fractal")
    for res in results:
        print(f"  - {res['text']}")
        
    # Fechar logger
    logger.close()
    print("\nTeste concluído")
    
if __name__ == "__main__":
    test_logger()