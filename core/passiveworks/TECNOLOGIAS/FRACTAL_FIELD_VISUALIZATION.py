import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, RegularPolygon
import math
import io
import base64
from PIL import Image

class FractalFieldVisualization:
    """
    Implementação de visualização de campos fractais
    para o sistema WiltonOS, demonstrando padrões Core-Shell-Orbit
    em diferentes níveis de escala.
    """
    
    def __init__(self):
        self.color_maps = {
            "galáxia": plt.cm.viridis,
            "célula": plt.cm.plasma,
            "molécula": plt.cm.cividis,
            "quântico": plt.cm.magma
        }
        
        self.scale_factors = {
            "galáxia": 1.0,
            "célula": 0.8,
            "molécula": 0.6,
            "quântico": 0.4
        }
    
    def generate_core_shell_orbit(self, scale="célula", complexity=5, resonance=0.7):
        """
        Gera uma visualização do padrão Core-Shell-Orbit em determinada escala
        
        Args:
            scale: Escala do padrão ("galáxia", "célula", "molécula", "quântico")
            complexity: Nível de complexidade (1-10)
            resonance: Harmonia entre as camadas (0-1)
        
        Returns:
            fig: Figura matplotlib com a visualização
        """
        # Configurar figura
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
        
        # Selecionar colormap baseado na escala
        cmap = self.color_maps.get(scale, plt.cm.viridis)
        scale_factor = self.scale_factors.get(scale, 1.0)
        
        # Limpar eixos
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        
        # Definir fundo
        background_color = '#0E1117'
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        
        # Gerar Core (núcleo)
        # Círculo central
        core_size = 0.2 * scale_factor
        core = Circle((0, 0), core_size, transform=ax.transData._b, 
                     color=cmap(0.8), alpha=0.9, zorder=10)
        ax.add_artist(core)
        
        # Gerar Shell (camada)
        shell_radius = 0.6 * scale_factor
        shell_width = 0.1 * scale_factor
        
        # Shell é representado como uma série de anéis
        num_shells = 3 + int(complexity * 0.5)
        for i in range(num_shells):
            r = shell_radius - (i * shell_width / num_shells)
            alpha = 0.7 * (1 - i/num_shells) * resonance
            shell = Circle((0, 0), r, transform=ax.transData._b,
                         color=cmap(0.6), alpha=alpha, fill=False, 
                         linewidth=2 * (1 - i/num_shells), zorder=9-i)
            ax.add_artist(shell)
        
        # Gerar Orbit (órbitas)
        # Representada como pontos/objetos orbitando em trajetórias
        num_orbits = 3 + int(complexity * 0.7)
        for i in range(num_orbits):
            # Raio da órbita
            orbit_radius = 0.3 + (i * 0.5 / num_orbits) * scale_factor
            
            # Número de objetos na órbita
            num_objects = 3 + i*2
            
            # Distribuir objetos na órbita
            theta = np.linspace(0, 2*np.pi, num_objects, endpoint=False)
            sizes = 0.05 * scale_factor * (0.8 + 0.4*np.random.random(num_objects))
            
            for j, angle in enumerate(theta):
                # Pequena variação aleatória no raio para tornar mais natural
                r_variation = orbit_radius * (1 + 0.1 * (np.random.random() - 0.5) * resonance)
                
                # Desenhar objeto orbital
                if scale == "galáxia":
                    # Estrelas para galáxia
                    x = r_variation * np.cos(angle)
                    y = r_variation * np.sin(angle)
                    color = cmap(0.2 + 0.6*np.random.random())
                    circle = Circle((x, y), sizes[j] * 0.5, transform=ax.transData._b,
                                  color=color, alpha=0.8, zorder=5)
                    ax.add_artist(circle)
                    
                elif scale == "célula":
                    # Organelas para célula
                    x = r_variation * np.cos(angle)
                    y = r_variation * np.sin(angle)
                    sides = 4 + j % 4
                    poly = RegularPolygon((x, y), sides, radius=sizes[j],
                                        orientation=np.random.random() * np.pi,
                                        transform=ax.transData._b,
                                        color=cmap(0.3 + 0.5*np.random.random()),
                                        alpha=0.7, zorder=5)
                    ax.add_artist(poly)
                    
                else:
                    # Pontos para outras escalas
                    x = r_variation * np.cos(angle)
                    y = r_variation * np.sin(angle)
                    circle = Circle((x, y), sizes[j], transform=ax.transData._b,
                                  color=cmap(0.4 + 0.4*np.random.random()),
                                  alpha=0.7, zorder=5)
                    ax.add_artist(circle)
        
        # Adicionar geometria sagrada sutil
        if complexity > 5:
            # Geometria sutil baseada em padrões da flor da vida
            radius_geo = 0.9 * scale_factor
            num_circles = 6
            for i in range(num_circles):
                angle = 2 * np.pi * i / num_circles
                x = 0.5 * radius_geo * np.cos(angle)
                y = 0.5 * radius_geo * np.sin(angle)
                circle = Circle((x, y), radius_geo * 0.5, transform=ax.transData._b,
                             fill=False, edgecolor=cmap(0.5), alpha=0.2,
                             linewidth=0.5, linestyle='--', zorder=1)
                ax.add_artist(circle)
        
        # Adicionar título
        scale_titles = {
            "galáxia": "Campo Fractal Galáctico",
            "célula": "Campo Fractal Celular",
            "molécula": "Campo Fractal Molecular",
            "quântico": "Campo Fractal Quântico"
        }
        
        plt.title(scale_titles.get(scale, "Campo Fractal"), 
                 color='white', fontsize=16, pad=20)
        
        # Definir limites
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        
        plt.tight_layout()
        return fig
    
    def generate_multi_scale_visualization(self, base_scale="célula", complexity=7):
        """
        Gera uma visualização multi-escala mostrando diferentes níveis
        do mesmo padrão fractal Core-Shell-Orbit
        
        Args:
            base_scale: Escala base para iniciar ("galáxia", "célula", etc)
            complexity: Nível de complexidade (1-10)
        
        Returns:
            fig: Figura matplotlib com visualização multi-escala
        """
        scales = ["galáxia", "célula", "molécula", "quântico"]
        
        # Criar figura com 4 subplots (2x2)
        fig, axs = plt.subplots(2, 2, figsize=(16, 16))
        fig.patch.set_facecolor('#0E1117')
        
        # Aplanar array de axes para fácil iteração
        axs = axs.flatten()
        
        # Título grande
        fig.suptitle("PADRÃO FRACTAL CORE-SHELL-ORBIT MULTI-ESCALA", 
                    color='white', fontsize=24, y=0.95)
        
        # Para cada escala
        for i, scale in enumerate(scales):
            ax = axs[i]
            ax.set_facecolor('#0E1117')
            
            # Camada atual
            self.draw_single_scale(ax, scale, complexity, use_polar=(i==0))
            
            # Adicionar título para esta escala
            scale_titles = {
                "galáxia": "Escala Galática",
                "célula": "Escala Celular",
                "molécula": "Escala Molecular",
                "quântico": "Escala Quântica"
            }
            ax.set_title(scale_titles.get(scale, "Escala Desconhecida"), 
                        color='white', fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig
    
    def draw_single_scale(self, ax, scale, complexity, use_polar=False):
        """
        Desenha um único padrão de escala em um eixo específico
        
        Args:
            ax: matplotlib axis para desenhar
            scale: Escala do padrão
            complexity: Complexidade (1-10)
            use_polar: Se deve usar coordenadas polares
        """
        # Configurar eixo
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Selecionar colormap e fator de escala
        cmap = self.color_maps.get(scale, plt.cm.viridis)
        scale_factor = self.scale_factors.get(scale, 1.0)
        
        # Core (núcleo)
        core_size = 0.15 * scale_factor
        core = Circle((0.5, 0.5), core_size, color=cmap(0.8), alpha=0.9, zorder=10)
        ax.add_artist(core)
        
        # Shell (camada)
        shell_radius = 0.4 * scale_factor
        shell_width = 0.08 * scale_factor
        
        # Shell como anéis
        num_shells = 3 + int(complexity * 0.5)
        for i in range(num_shells):
            r = shell_radius - (i * shell_width / num_shells)
            alpha = 0.7 * (1 - i/num_shells)
            shell = Circle((0.5, 0.5), r, color=cmap(0.6), alpha=alpha, fill=False, 
                         linewidth=2 * (1 - i/num_shells), zorder=9-i)
            ax.add_artist(shell)
        
        # Orbit (órbitas)
        num_orbits = 2 + int(complexity * 0.7)
        for i in range(num_orbits):
            # Raio da órbita
            orbit_radius = 0.2 + (i * 0.3 / num_orbits) * scale_factor
            
            # Objetos orbitando
            num_objects = 3 + i*2
            theta = np.linspace(0, 2*np.pi, num_objects, endpoint=False)
            
            for angle in theta:
                # Posição do objeto
                x = 0.5 + orbit_radius * np.cos(angle)
                y = 0.5 + orbit_radius * np.sin(angle)
                
                # Tamanho do objeto
                size = 0.04 * scale_factor * (0.8 + 0.4*np.random.random())
                
                # Estilo baseado na escala
                if scale == "galáxia":
                    color = cmap(0.2 + 0.6*np.random.random())
                    circle = Circle((x, y), size * 0.5, color=color, alpha=0.8, zorder=5)
                    ax.add_artist(circle)
                elif scale == "célula":
                    sides = 4 + int(angle * 3 / np.pi) % 4
                    poly = RegularPolygon((x, y), sides, radius=size,
                                        orientation=np.random.random() * np.pi,
                                        color=cmap(0.3 + 0.5*np.random.random()),
                                        alpha=0.7, zorder=5)
                    ax.add_artist(poly)
                else:
                    circle = Circle((x, y), size, color=cmap(0.4 + 0.4*np.random.random()),
                                  alpha=0.7, zorder=5)
                    ax.add_artist(circle)
        
        # Adicionar geometria sagrada sutil
        if complexity > 5:
            radius_geo = 0.45 * scale_factor
            num_circles = 6
            for i in range(num_circles):
                angle = 2 * np.pi * i / num_circles
                x = 0.5 + 0.25 * radius_geo * np.cos(angle)
                y = 0.5 + 0.25 * radius_geo * np.sin(angle)
                circle = Circle((x, y), radius_geo, fill=False, edgecolor=cmap(0.5), 
                             alpha=0.15, linewidth=0.5, linestyle='--', zorder=1)
                ax.add_artist(circle)
        
        # Definir limites
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
    
    def save_visualization(self, fig, filename="fractal_field.png"):
        """Salva a visualização como imagem"""
        fig.savefig(filename, facecolor=fig.get_facecolor(), dpi=100, bbox_inches='tight')
        return filename
    
    def get_image_base64(self, fig):
        """Converte figura para base64 para uso em HTML/Streamlit"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str

def add_to_streamlit():
    """Interface Streamlit para o visualizador"""
    st.title("Visualizador de Campo Fractal")
    
    st.markdown("""
    > "De capsídeos ao cosmos, a mesma gramática estrutural se repete."
    
    Este visualizador demonstra o padrão Core-Shell-Orbit em diferentes escalas, 
    mostrando a natureza fractal da realidade através da lente WiltonOS.
    """)
    
    # Inicializar visualizador
    ffv = FractalFieldVisualization()
    
    # Parâmetros de configuração
    st.sidebar.header("Configurações")
    
    tab1, tab2 = st.tabs(["Visualização Única", "Multi-Escala"])
    
    with tab1:
        scale = st.selectbox(
            "Escala",
            ["célula", "galáxia", "molécula", "quântico"],
            key="scale_single"
        )
        
        complexity = st.slider(
            "Complexidade",
            1, 10, 7,
            key="complexity_single"
        )
        
        resonance = st.slider(
            "Ressonância",
            0.0, 1.0, 0.7,
            key="resonance"
        )
        
        if st.button("Gerar Visualização", key="gen_single"):
            with st.spinner("Gerando campo fractal..."):
                fig = ffv.generate_core_shell_orbit(scale, complexity, resonance)
                st.pyplot(fig)
                
                # Opção para salvar
                img_data = ffv.get_image_base64(fig)
                st.markdown(f"### Download")
                href = f'<a href="data:image/png;base64,{img_data}" download="campo_fractal_{scale}.png">Clique para baixar imagem</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    with tab2:
        base_scale = st.selectbox(
            "Escala Base",
            ["célula", "galáxia", "molécula", "quântico"],
            key="scale_multi"
        )
        
        complexity_multi = st.slider(
            "Complexidade",
            1, 10, 7,
            key="complexity_multi"
        )
        
        if st.button("Gerar Visualização Multi-Escala", key="gen_multi"):
            with st.spinner("Gerando campos fractais multi-escala..."):
                fig = ffv.generate_multi_scale_visualization(base_scale, complexity_multi)
                st.pyplot(fig)
                
                # Opção para salvar
                img_data = ffv.get_image_base64(fig)
                st.markdown(f"### Download")
                href = f'<a href="data:image/png;base64,{img_data}" download="campos_fractais_multi_escala.png">Clique para baixar imagem</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    st.sidebar.markdown("## Referência")
    st.sidebar.markdown("""
    Este visualizador implementa os princípios descritos em `FRACTAL_FIELD_REF_001.md`:
    
    1. **CORE** - Núcleo central de processamento/identidade
    2. **SHELL** - Camada de interface/tradução
    3. **ORBIT** - Componentes especializados em trajetórias funcionais
    """)

if __name__ == "__main__":
    add_to_streamlit()