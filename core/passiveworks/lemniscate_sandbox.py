import os
import time
import tempfile
import json
from datetime import datetime
import io

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from openai import OpenAI

class LemniscateSandbox:
    def __init__(self):
        self.stage = "REFLECTION"  # REFLECTION, CREATION, TRANSFORMATION
        self.logs = []
        self.client = None
        
        # Verificar API key
        if "OPENAI_API_KEY" in os.environ:
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self.log("API OpenAI configurada com sucesso.", "info")
        else:
            self.log("API key OpenAI não encontrada. Algumas funcionalidades estarão limitadas.", "warning")
    
    def log(self, message, log_type="info"):
        """Adiciona uma mensagem ao log"""
        self.logs.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "message": message,
            "type": log_type
        })
    
    def set_stage(self, stage):
        """Define o estágio atual do Lemniscate"""
        if stage in ["REFLECTION", "CREATION", "TRANSFORMATION"]:
            self.stage = stage
            self.log(f"Estágio alterado para {stage}", "info")
            return True
        return False
    
    def draw_lemniscate(self):
        """Desenha a visualização da lemniscata com o estágio atual"""
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Parâmetros para a lemniscata
        a = 1
        t = np.linspace(0, 2*np.pi, 1000)
        x = a * np.cos(t) / (1 + np.sin(t)**2)
        y = a * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
        
        # Plotar a lemniscata
        ax.plot(x, y, 'k-', linewidth=2)
        
        # Marcar os pontos específicos
        points = {
            "REFLECTION": (0, 0),
            "CREATION": (a, 0),
            "TRANSFORMATION": (-a, 0)
        }
        
        # Cores para cada estágio
        colors = {
            "REFLECTION": '#42A5F5',  # Azul
            "CREATION": '#66BB6A',    # Verde
            "TRANSFORMATION": '#FF7043'  # Laranja
        }
        
        # Marcar todos os pontos
        for s, (px, py) in points.items():
            if s == self.stage:
                ax.scatter(px, py, color=colors[s], s=150, zorder=5, label=s)
            else:
                ax.scatter(px, py, color=colors[s], s=80, alpha=0.5, zorder=4, label=s)
        
        # Adicionar anotações
        for s, (px, py) in points.items():
            y_offset = 0.2 if s == "REFLECTION" else 0.3
            ax.annotate(s, (px, py + y_offset), ha='center', fontsize=12, color=colors[s])
        
        # Adicionar explicações
        explanations = {
            "REFLECTION": "Observar e Avaliar",
            "CREATION": "Idealizar e Manifestar",
            "TRANSFORMATION": "Mudar e Transcender"
        }
        
        # Mostrar a explicação do estágio atual
        fig.text(0.5, 0.05, explanations[self.stage], ha='center', fontsize=14, color=colors[self.stage])
        
        # Personalização
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1, 1)
        ax.axis('off')
        plt.tight_layout()
        
        # Converter para imagem
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        buf.seek(0)
        img = Image.open(buf)
        
        return img
    
    def get_stage_description(self):
        """Retorna a descrição do estágio atual"""
        descriptions = {
            "REFLECTION": """
            **REFLEXÃO** é o espaço de observação e avaliação.
            
            Neste estágio, você:
            - Observa padrões emergentes
            - Avalia experiências passadas
            - Cria distância para ganhar clareza
            - Sintetiza aprendizados
            
            É o momento de **receber** e **contemplar**.
            """,
            
            "CREATION": """
            **CRIAÇÃO** é o espaço de manifestação e idealização.
            
            Neste estágio, você:
            - Transforma ideias em formas
            - Concretiza insights em ações
            - Constrói estruturas e sistemas
            - Traz o interno para o externo
            
            É o momento de **expressar** e **materializar**.
            """,
            
            "TRANSFORMATION": """
            **TRANSFORMAÇÃO** é o espaço de mudança e transcendência.
            
            Neste estágio, você:
            - Dissolve formas limitantes
            - Integra polaridades
            - Transmuta energia bloqueada
            - Expande para novas possibilidades
            
            É o momento de **soltar** e **evoluir**.
            """
        }
        
        return descriptions[self.stage]
    
    def create_sacred_geometry(self, type="flower_of_life"):
        """Cria uma visualização de geometria sagrada"""
        img = Image.new('RGBA', (1000, 1000), (30, 30, 40, 255))
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = 500, 500
        
        if type == "flower_of_life":
            radius = 150
            # Criar a Flor da Vida
            for angle in range(0, 360, 60):
                x = center_x + radius * np.cos(np.radians(angle))
                y = center_y + radius * np.sin(np.radians(angle))
                
                # Desenhar círculo
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=(100, 200, 255, 200), width=2)
            
            # Círculo central
            draw.ellipse((center_x-radius, center_y-radius, center_x+radius, center_y+radius), 
                        outline=(100, 200, 255, 200), width=2)
        
        elif type == "metatron_cube":
            radius = 300
            points = []
            
            # Criar os pontos do cubo de Metatron
            for angle in range(0, 360, 60):
                x = center_x + radius * np.cos(np.radians(angle))
                y = center_y + radius * np.sin(np.radians(angle))
                points.append((x, y))
                
                # Ponto interno
                x_in = center_x + radius/2 * np.cos(np.radians(angle))
                y_in = center_y + radius/2 * np.sin(np.radians(angle))
                points.append((x_in, y_in))
            
            # Adicionar ponto central
            points.append((center_x, center_y))
            
            # Desenhar linhas
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    draw.line([points[i], points[j]], fill=(100, 200, 255, 150), width=2)
        
        elif type == "sri_yantra":
            radius = 400
            
            # Triângulos apontando para cima
            for i in range(5):
                scale = 0.4 + i * 0.12
                points = [
                    (center_x, center_y - radius * scale),
                    (center_x - radius * scale * 0.866, center_y + radius * scale * 0.5),
                    (center_x + radius * scale * 0.866, center_y + radius * scale * 0.5)
                ]
                draw.polygon(points, outline=(255, 100, 100, 200), fill=(0, 0, 0, 0))
            
            # Triângulos apontando para baixo
            for i in range(4):
                scale = 0.46 + i * 0.12
                points = [
                    (center_x, center_y + radius * scale),
                    (center_x - radius * scale * 0.866, center_y - radius * scale * 0.5),
                    (center_x + radius * scale * 0.866, center_y - radius * scale * 0.5)
                ]
                draw.polygon(points, outline=(100, 200, 255, 200), fill=(0, 0, 0, 0))
            
            # Círculos
            for i in range(3):
                r = radius * (0.2 + i * 0.2)
                draw.ellipse((center_x-r, center_y-r, center_x+r, center_y+r), 
                             outline=(200, 200, 200, 200), width=2)
        
        # Salvar imagem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"geometria_{type}_{timestamp}.png"
        img.save(filename)
        
        return filename, img
    
    def generate_framework(self, concept1, concept2=None):
        """Gera um framework baseado em um ou dois conceitos"""
        if not self.client:
            self.log("Não é possível gerar framework sem a API OpenAI configurada.", "error")
            return "API OpenAI não configurada", None
        
        try:
            prompt = f"Crie um framework estruturado para o conceito de {concept1}"
            if concept2:
                prompt = f"Crie um framework integrativo que sintetize os conceitos de {concept1} e {concept2}"
            
            stage_prompts = {
                "REFLECTION": "Enfatize aspectos de observação, análise e avaliação.",
                "CREATION": "Enfatize aspectos de manifestação, construção e implementação.",
                "TRANSFORMATION": "Enfatize aspectos de mudança, integração e transcendência."
            }
            
            prompt += f" {stage_prompts[self.stage]} O framework deve incluir: 1) Definição conceitual, 2) Princípios fundamentais, 3) Metodologia, 4) Aplicações práticas, 5) Métricas de avaliação. Formato em markdown."
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Você é um especialista em criar frameworks conceituais que integram conhecimento multidisciplinar. Seu objetivo é criar estruturas claras, práticas e profundas que possam guiar a implementação de conceitos complexos."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            framework = response.choices[0].message.content
            
            # Salvar framework em arquivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"framework_{concept1.replace(' ', '_')}_{timestamp}.md"
            
            with open(filename, "w") as f:
                f.write(framework)
            
            return framework, filename
        
        except Exception as e:
            self.log(f"Erro ao gerar framework: {str(e)}", "error")
            return f"Erro ao gerar framework: {str(e)}", None
    
    def analyze_document(self, document_content):
        """Analisa um documento e extrai insights principais"""
        if not self.client:
            self.log("Não é possível analisar documento sem a API OpenAI configurada.", "error")
            return "API OpenAI não configurada"
        
        try:
            stage_prompts = {
                "REFLECTION": "Foque em observações, padrões e insights profundos.",
                "CREATION": "Foque em oportunidades de implementação e manifestação.",
                "TRANSFORMATION": "Foque em potenciais de transformação e transcendência."
            }
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"Você é um analista especializado em extrair insights profundos de documentos. {stage_prompts[self.stage]} Estruture sua análise de forma clara e acionável, identificando pontos principais, conexões ocultas e próximos passos."},
                    {"role": "user", "content": f"Analise este documento e extraia os insights mais relevantes:\n\n{document_content[:3000]}..."}
                ],
                temperature=0.6,
                max_tokens=1000
            )
            
            analysis = response.choices[0].message.content
            
            # Salvar análise em arquivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analise_documento_{timestamp}.md"
            
            with open(filename, "w") as f:
                f.write(f"# Análise de Documento\n\n")
                f.write(f"*Gerada em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}*\n\n")
                f.write(f"*Estágio Lemniscate: {self.stage}*\n\n")
                f.write(analysis)
            
            return analysis, filename
        
        except Exception as e:
            self.log(f"Erro ao analisar documento: {str(e)}", "error")
            return f"Erro ao analisar documento: {str(e)}", None
    
    def transmute_challenge(self, challenge):
        """Transmuta um desafio em oportunidade"""
        if not self.client:
            self.log("Não é possível transmutar desafio sem a API OpenAI configurada.", "error")
            return "API OpenAI não configurada"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Você é um especialista em transmutação de energia e perspectiva. Sua habilidade é transformar desafios em oportunidades de crescimento, revelando o potencial positivo escondido nas dificuldades. Utilize linguagem poética, metafórica e inspiradora, mas mantenha o conteúdo prático e aplicável."},
                    {"role": "user", "content": f"Transmute este desafio em uma oportunidade de crescimento e evolução: {challenge}"}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            transmutation = response.choices[0].message.content
            
            # Salvar transmutação em arquivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transmutacao_{timestamp}.md"
            
            with open(filename, "w") as f:
                f.write(f"# Transmutação de Desafio\n\n")
                f.write(f"*Gerada em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}*\n\n")
                f.write(f"**Desafio Original:** {challenge}\n\n")
                f.write(transmutation)
            
            return transmutation, filename
        
        except Exception as e:
            self.log(f"Erro ao transmutar desafio: {str(e)}", "error")
            return f"Erro ao transmutar desafio: {str(e)}", None
    
    def integrate_polarities(self, polarity1, polarity2):
        """Integra polaridades aparentemente opostas"""
        if not self.client:
            self.log("Não é possível integrar polaridades sem a API OpenAI configurada.", "error")
            return "API OpenAI não configurada"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Você é um especialista em integração de polaridades, capaz de encontrar a síntese transcendente entre opostos aparentes. Utilize conceitos de dialética, não-dualidade e sistemas complexos para revelar como os opostos são na verdade aspectos complementares de uma realidade mais ampla."},
                    {"role": "user", "content": f"Crie um framework para integrar as polaridades de {polarity1} e {polarity2} em uma síntese transcendente que honre a verdade de ambos os polos."}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            integration = response.choices[0].message.content
            
            # Salvar integração em arquivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"integracao_{polarity1}_{polarity2}_{timestamp}.md"
            
            with open(filename, "w") as f:
                f.write(f"# Integração de Polaridades: {polarity1} & {polarity2}\n\n")
                f.write(f"*Gerada em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}*\n\n")
                f.write(integration)
            
            return integration, filename
        
        except Exception as e:
            self.log(f"Erro ao integrar polaridades: {str(e)}", "error")
            return f"Erro ao integrar polaridades: {str(e)}", None

# Função para instanciar e retornar o objeto LemniscateSandbox
def get_lemniscate_sandbox():
    if 'lemniscate_sandbox_instance' not in st.session_state:
        st.session_state.lemniscate_sandbox_instance = LemniscateSandbox()
    
    return st.session_state.lemniscate_sandbox_instance