import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import datetime
import io
import base64
from PIL import Image
import json
import os

class FractalDecayRitual:
    """
    Implementação do Ritual de Decaimento Fractal
    para o sistema WiltonOS, permitindo o registro e 
    tracking de entidades em processo de decaimento
    com preservação de memória de campo.
    """
    
    def __init__(self):
        self.today = datetime.datetime.now().date()
        
        # Categorias de entidades
        self.entity_categories = [
            "Pessoa",
            "Animal",
            "Lugar",
            "Objeto",
            "Projeto",
            "Conceito",
            "Relação",
            "Outro"
        ]
        
        # Estados de decaimento
        self.decay_states = [
            "Recente (0-6 meses)",
            "Ativo (6 meses - 2 anos)",
            "Transição (2-5 anos)",
            "Transformação (5-10 anos)",
            "Campo Residual (10+ anos)"
        ]
        
        # Tipos de half-life emocional/memorial
        self.half_life_types = [
            "Ultra Rápido (dias)",
            "Rápido (semanas)",
            "Médio (meses)",
            "Lento (1-2 anos)",
            "Muito Lento (3-5 anos)",
            "Legacy (10+ anos)"
        ]
        
        # Mapeamento para valores reais de half-life em dias
        self.half_life_values = {
            "Ultra Rápido (dias)": 7,
            "Rápido (semanas)": 30,
            "Médio (meses)": 90,
            "Lento (1-2 anos)": 365,
            "Muito Lento (3-5 anos)": 1095,
            "Legacy (10+ anos)": 3650
        }
        
        # Carregar dados existentes ou criar novo dataframe
        self.data_file = "TECNOLOGIAS/fractal_decay_entities.json"
        self.load_data()
    
    def load_data(self):
        """Carregar dados existentes ou criar nova estrutura"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.entities = data
            else:
                self.entities = []
        except Exception as e:
            st.error(f"Erro ao carregar dados: {str(e)}")
            self.entities = []
    
    def save_data(self):
        """Salvar dados no arquivo JSON"""
        try:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.entities, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            st.error(f"Erro ao salvar dados: {str(e)}")
            return False
    
    def calculate_decay_curve(self, initial_value, half_life, days=1000):
        """
        Calcula a curva de decaimento baseada na fórmula de half-life
        
        Args:
            initial_value: Valor inicial (100%)
            half_life: Valor de meia-vida em dias
            days: Número de dias para calcular
            
        Returns:
            Tupla com arrays de dias e valores
        """
        time_points = np.linspace(0, days, 100)
        values = initial_value * np.power(0.5, time_points / half_life)
        return time_points, values
    
    def calculate_revisit_points(self, registration_date, half_life, cycles=5):
        """
        Calcula pontos de revisitação baseados em meia-vida
        
        Args:
            registration_date: Data de registro inicial
            half_life: Valor de meia-vida em dias
            cycles: Número de ciclos para calcular
            
        Returns:
            Lista de datas para reconsolidação
        """
        revisit_points = []
        
        # Converter data de string para objeto de data
        if isinstance(registration_date, str):
            start_date = datetime.datetime.strptime(registration_date, "%Y-%m-%d").date()
        else:
            start_date = registration_date
        
        # Calcular pontos exponenciais (1x, 2x, 4x, 8x de meia-vida)
        for i in range(cycles):
            days_to_add = half_life * (2**i)
            revisit_date = start_date + datetime.timedelta(days=days_to_add)
            revisit_points.append({
                "cycle": i+1,
                "days_from_start": days_to_add,
                "date": revisit_date.strftime("%Y-%m-%d"),
                "percentage": 100 * (0.5**(i+1)),
                "days_remaining": (revisit_date - self.today).days
            })
        
        return revisit_points
    
    def register_entity(self, 
                        name, 
                        category, 
                        core_essence, 
                        primary_connections,
                        recognition_signals,
                        decay_state,
                        half_life_type,
                        registration_date=None,
                        anchoring_artifacts=None,
                        notes=None):
        """
        Registra uma nova entidade no sistema de decaimento fractal
        
        Args:
            name: Nome ou identificação da entidade
            category: Categoria (pessoa, animal, lugar, etc)
            core_essence: Essência/qualidade core
            primary_connections: Conexões primárias
            recognition_signals: Sinais de reconhecimento
            decay_state: Estado atual de decaimento
            half_life_type: Tipo de half-life emocional/memorial
            registration_date: Data de registro (padrão: hoje)
            anchoring_artifacts: Artefatos de ancoragem (opcional)
            notes: Notas adicionais (opcional)
            
        Returns:
            ID da entidade registrada
        """
        if registration_date is None:
            registration_date = self.today.strftime("%Y-%m-%d")
            
        half_life_value = self.half_life_values.get(half_life_type, 365)  # Padrão: 1 ano
            
        # Calcular pontos de revisitação
        revisit_points = self.calculate_revisit_points(
            registration_date, 
            half_life_value,
            cycles=5
        )
            
        # Criar registro da entidade
        entity = {
            "id": len(self.entities) + 1,
            "name": name,
            "category": category,
            "core_essence": core_essence,
            "primary_connections": primary_connections,
            "recognition_signals": recognition_signals,
            "decay_state": decay_state,
            "half_life_type": half_life_type,
            "half_life_value": half_life_value,
            "registration_date": registration_date,
            "anchoring_artifacts": anchoring_artifacts if anchoring_artifacts else [],
            "notes": notes if notes else "",
            "revisit_points": revisit_points
        }
            
        # Adicionar à lista de entidades
        self.entities.append(entity)
            
        # Salvar dados
        self.save_data()
            
        return entity["id"]
    
    def update_entity(self, entity_id, field, value):
        """
        Atualiza um campo de uma entidade existente
        
        Args:
            entity_id: ID da entidade
            field: Campo a atualizar
            value: Novo valor
            
        Returns:
            Boolean indicando sucesso
        """
        for i, entity in enumerate(self.entities):
            if entity["id"] == entity_id:
                # Se o campo for half_life_type, recalcular pontos de revisitação
                if field == "half_life_type":
                    half_life_value = self.half_life_values.get(value, 365)
                    self.entities[i]["half_life_value"] = half_life_value
                    self.entities[i]["revisit_points"] = self.calculate_revisit_points(
                        self.entities[i]["registration_date"],
                        half_life_value,
                        cycles=5
                    )
                
                # Atualizar o campo
                self.entities[i][field] = value
                self.save_data()
                return True
        
        return False
    
    def delete_entity(self, entity_id):
        """
        Remove uma entidade do sistema
        
        Args:
            entity_id: ID da entidade a remover
            
        Returns:
            Boolean indicando sucesso
        """
        for i, entity in enumerate(self.entities):
            if entity["id"] == entity_id:
                self.entities.pop(i)
                self.save_data()
                return True
        
        return False
    
    def get_entity(self, entity_id):
        """
        Recupera uma entidade pelo ID
        
        Args:
            entity_id: ID da entidade
            
        Returns:
            Dicionário com dados da entidade ou None
        """
        for entity in self.entities:
            if entity["id"] == entity_id:
                return entity
        
        return None
    
    def get_next_revisits(self, days_ahead=90, max_items=5):
        """
        Recupera próximas datas de revisitação em ordem cronológica
        
        Args:
            days_ahead: Número de dias a considerar
            max_items: Número máximo de itens a retornar
            
        Returns:
            Lista de dicionários com entidades e datas
        """
        upcoming_revisits = []
        
        for entity in self.entities:
            for point in entity["revisit_points"]:
                if 0 < point["days_remaining"] <= days_ahead:
                    upcoming_revisits.append({
                        "entity_id": entity["id"],
                        "entity_name": entity["name"],
                        "entity_category": entity["category"],
                        "revisit_date": point["date"],
                        "days_remaining": point["days_remaining"],
                        "cycle": point["cycle"],
                        "percentage": point["percentage"]
                    })
        
        # Ordenar por dias restantes
        upcoming_revisits.sort(key=lambda x: x["days_remaining"])
        
        # Limitar ao número máximo
        return upcoming_revisits[:max_items]
    
    def plot_decay_curve(self, entity_id):
        """
        Gera uma visualização da curva de decaimento para uma entidade
        
        Args:
            entity_id: ID da entidade
            
        Returns:
            Figura matplotlib
        """
        entity = self.get_entity(entity_id)
        
        if not entity:
            return None
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        
        # Configurar eixos
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#CCCCCC')
        ax.grid(alpha=0.2, linestyle='--')
        
        # Calcular curva de decaimento
        days, values = self.calculate_decay_curve(100, entity["half_life_value"], days=entity["half_life_value"] * 10)
        
        # Plotar curva principal
        ax.plot(days, values, 'r-', linewidth=2, color='#FF5252', label='Intensidade de Memória')
        
        # Adicionar linha horizontal em 50%
        ax.axhline(y=50, color='#CCCCCC', linestyle='--', alpha=0.5)
        ax.text(entity["half_life_value"] * 1.05, 50, '50% (Half-Life)', color='#CCCCCC')
        
        # Adicionar linha vertical no ponto de half-life
        ax.axvline(x=entity["half_life_value"], color='#CCCCCC', linestyle='--', alpha=0.5)
        
        # Adicionar pontos de revisitação
        revisit_days = [point["days_from_start"] for point in entity["revisit_points"]]
        revisit_values = [100 * (0.5**(i+1)) for i in range(len(revisit_days))]
        
        ax.scatter(revisit_days, revisit_values, color='#42A5F5', s=80, zorder=5, label='Pontos de Reconsolidação')
        
        # Adicionar rótulos para os pontos
        for i, (day, value) in enumerate(zip(revisit_days, revisit_values)):
            ax.text(day, value + 3, f'Ciclo {i+1}', color='#FFFFFF', ha='center')
        
        # Adicionar registro inicial
        ax.scatter([0], [100], color='#66BB6A', s=100, zorder=5, label='Registro Inicial')
        
        # Adicionar data atual
        registration_date = datetime.datetime.strptime(entity["registration_date"], "%Y-%m-%d").date()
        days_since_registration = (self.today - registration_date).days
        
        if days_since_registration > 0 and days_since_registration < max(days):
            current_value = 100 * (0.5**(days_since_registration / entity["half_life_value"]))
            ax.scatter([days_since_registration], [current_value], color='#FFEB3B', s=100, zorder=5, label='Posição Atual')
            ax.axvline(x=days_since_registration, color='#FFEB3B', linestyle='--', alpha=0.3)
        
        # Configurar rótulos
        ax.set_xlabel('Dias', color='#FFFFFF', fontsize=12)
        ax.set_ylabel('Intensidade de Memória (%)', color='#FFFFFF', fontsize=12)
        ax.set_title(f'Curva de Decaimento Fractal: {entity["name"]}', color='#FFFFFF', fontsize=14)
        
        # Adicionar legenda
        ax.legend(framealpha=0.7, facecolor='#1A1E24', edgecolor='#CCCCCC', fontsize=10)
        
        # Evitar valores negativos e limitar o eixo y
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        return fig
    
    def plot_entity_timeline(self, entity_id):
        """
        Gera uma visualização do timeline de revisitação para uma entidade
        
        Args:
            entity_id: ID da entidade
            
        Returns:
            Figura matplotlib
        """
        entity = self.get_entity(entity_id)
        
        if not entity:
            return None
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        
        # Configurar eixos
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', colors='#CCCCCC')
        ax.set_yticks([])
        
        # Obter datas
        registration_date = datetime.datetime.strptime(entity["registration_date"], "%Y-%m-%d").date()
        revisit_dates = [datetime.datetime.strptime(point["date"], "%Y-%m-%d").date() for point in entity["revisit_points"]]
        
        # Criar timeline
        dates = [registration_date] + revisit_dates
        labels = ["Registro"] + [f"Ciclo {i+1}" for i in range(len(revisit_dates))]
        positions = list(range(len(dates)))
        
        # Plotar pontos
        ax.scatter(positions, [0] * len(positions), s=200, color=['#66BB6A'] + ['#42A5F5'] * len(revisit_dates), zorder=3)
        
        # Conectar pontos com linha
        ax.plot(positions, [0] * len(positions), 'o-', color='#CCCCCC', alpha=0.5, linewidth=2, markersize=0)
        
        # Adicionar rótulos de data
        for i, (pos, date) in enumerate(zip(positions, dates)):
            ax.annotate(date.strftime("%d/%m/%Y"), 
                     xy=(pos, 0),
                     xytext=(0, 20),
                     textcoords="offset points",
                     ha='center',
                     color='#FFFFFF',
                     fontsize=10)
            
            ax.annotate(labels[i], 
                     xy=(pos, 0),
                     xytext=(0, -20),
                     textcoords="offset points",
                     ha='center',
                     color='#FFFFFF',
                     fontweight='bold',
                     fontsize=10)
        
        # Adicionar posição atual
        days_since_registration = (self.today - registration_date).days
        
        if days_since_registration > 0:
            # Converter dias para posição no timeline
            full_range = (dates[-1] - dates[0]).days
            relative_position = days_since_registration / full_range * (len(positions) - 1)
            
            # Adicionar marcador para posição atual
            ax.scatter([relative_position], [0], s=150, color='#FFEB3B', zorder=4, marker='D')
            ax.annotate("Hoje", 
                      xy=(relative_position, 0),
                      xytext=(0, 40),
                      textcoords="offset points",
                      ha='center',
                      color='#FFEB3B',
                      fontweight='bold',
                      fontsize=10)
        
        # Configurar rótulos
        ax.set_title(f'Timeline de Reconsolidação: {entity["name"]}', color='#FFFFFF', fontsize=14)
        
        # Configurar limites
        ax.set_xlim(-0.5, len(positions) - 0.5)
        ax.set_ylim(-1, 1)
        
        plt.tight_layout()
        return fig
    
    def render_ui(self):
        """Interface principal do Ritual de Decaimento Fractal"""
        st.subheader("🧬 Ritual de Decaimento Fractal")
        
        st.markdown("""
        > *"Decay is not death — it's rhythmic transformation."*
        
        Este módulo implementa o conceito de *Decaimento Preservando Memória de Campo*, 
        permitindo o registro e acompanhamento de entidades em processo de transformação.
        """)
        
        # Tabs para diferentes funcionalidades
        tab1, tab2, tab3, tab4 = st.tabs([
            "📝 Registro de Entidade", 
            "🔍 Visualizar Entidades", 
            "📊 Próximos Ciclos",
            "📚 Documentação"
        ])
        
        # Tab 1: Registro de Entidade
        with tab1:
            st.markdown("### Registro de Nova Entidade")
            st.markdown("Registre uma entidade (pessoa, animal, lugar, etc) no sistema de decaimento fractal.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Nome/Identificação", key="reg_name")
                category = st.selectbox("Categoria", self.entity_categories, key="reg_category")
                core_essence = st.text_area("Essência/Qualidade Core", 
                                         placeholder="Descreva a essência fundamental desta entidade",
                                         help="Características definidoras que persistem mesmo após transformação",
                                         key="reg_core")
                primary_connections = st.text_area("Conexões Primárias", 
                                                placeholder="Relações e conexões principais",
                                                help="Entidades, conceitos ou contextos conectados",
                                                key="reg_connections")
            
            with col2:
                recognition_signals = st.text_area("Sinais de Reconhecimento", 
                                                placeholder="Como reconhecer esta entidade",
                                                help="Traços, sinais ou gatilhos que ativam a memória",
                                                key="reg_signals")
                decay_state = st.selectbox("Estado de Decaimento", self.decay_states, key="reg_state")
                half_life_type = st.selectbox("Tipo de Half-Life", self.half_life_types, key="reg_halflife")
                
                registration_date = st.date_input("Data de Registro", value=self.today, key="reg_date")
            
            st.markdown("#### Artefatos de Ancoragem (opcional)")
            st.markdown("Objetos físicos ou práticas que servem como pontos de acesso à memória.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                artifact1 = st.text_input("Artefato 1", key="reg_art1")
            with col2:
                artifact2 = st.text_input("Artefato 2", key="reg_art2")
            with col3:
                artifact3 = st.text_input("Artefato 3", key="reg_art3")
            
            notes = st.text_area("Notas Adicionais", key="reg_notes")
            
            if st.button("Registrar Entidade", use_container_width=True):
                if not name:
                    st.error("Nome/Identificação é obrigatório.")
                else:
                    # Recolher artefatos não vazios
                    artifacts = []
                    for art in [artifact1, artifact2, artifact3]:
                        if art:
                            artifacts.append(art)
                    
                    # Registrar entidade
                    entity_id = self.register_entity(
                        name=name,
                        category=category,
                        core_essence=core_essence,
                        primary_connections=primary_connections,
                        recognition_signals=recognition_signals,
                        decay_state=decay_state,
                        half_life_type=half_life_type,
                        registration_date=registration_date.strftime("%Y-%m-%d"),
                        anchoring_artifacts=artifacts,
                        notes=notes
                    )
                    
                    st.success(f"Entidade '{name}' registrada com sucesso!")
                    
                    # Mostrar cronograma de revisitação
                    if entity_id:
                        entity = self.get_entity(entity_id)
                        st.markdown("### Cronograma de Reconsolidação")
                        st.markdown("Pontos de revisitação para reconsolidação de memória:")
                        
                        revisit_df = pd.DataFrame(entity["revisit_points"])
                        revisit_df = revisit_df[["cycle", "date", "percentage", "days_remaining"]]
                        revisit_df.columns = ["Ciclo", "Data", "Intensidade (%)", "Dias Restantes"]
                        
                        st.dataframe(revisit_df)
                        
                        # Mostrar curva de decaimento
                        st.markdown("### Curva de Decaimento")
                        fig = self.plot_decay_curve(entity_id)
                        st.pyplot(fig)
        
        # Tab 2: Visualizar Entidades
        with tab2:
            st.markdown("### Entidades Registradas")
            
            if not self.entities:
                st.info("Nenhuma entidade registrada ainda. Use a aba 'Registro de Entidade' para começar.")
            else:
                # Filtro por categoria
                category_filter = st.selectbox(
                    "Filtrar por Categoria", 
                    ["Todas"] + self.entity_categories,
                    key="vis_category"
                )
                
                # Filtrar entidades
                filtered_entities = self.entities
                if category_filter != "Todas":
                    filtered_entities = [e for e in self.entities if e["category"] == category_filter]
                
                # Mostrar lista de entidades
                entity_names = [e["name"] for e in filtered_entities]
                selected_entity_name = st.selectbox("Selecionar Entidade", entity_names, key="vis_entity")
                
                # Obter entidade selecionada
                selected_entity = next((e for e in filtered_entities if e["name"] == selected_entity_name), None)
                
                if selected_entity:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"### {selected_entity['name']}")
                        st.markdown(f"**Categoria:** {selected_entity['category']}")
                        st.markdown(f"**Registrado em:** {selected_entity['registration_date']}")
                        st.markdown(f"**Estado de Decaimento:** {selected_entity['decay_state']}")
                        st.markdown(f"**Tipo de Half-Life:** {selected_entity['half_life_type']}")
                        
                        st.markdown("#### Essência/Qualidade Core")
                        st.markdown(selected_entity['core_essence'])
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.markdown("#### Conexões Primárias")
                            st.markdown(selected_entity['primary_connections'])
                        
                        with col_b:
                            st.markdown("#### Sinais de Reconhecimento")
                            st.markdown(selected_entity['recognition_signals'])
                        
                        if selected_entity['anchoring_artifacts']:
                            st.markdown("#### Artefatos de Ancoragem")
                            for artifact in selected_entity['anchoring_artifacts']:
                                st.markdown(f"- {artifact}")
                        
                        if selected_entity['notes']:
                            st.markdown("#### Notas")
                            st.markdown(selected_entity['notes'])
                    
                    with col2:
                        st.markdown("#### Cronograma de Reconsolidação")
                        
                        revisit_df = pd.DataFrame(selected_entity["revisit_points"])
                        revisit_df = revisit_df[["cycle", "date", "percentage", "days_remaining"]]
                        revisit_df.columns = ["Ciclo", "Data", "Intens. (%)", "Dias"]
                        
                        st.dataframe(revisit_df)
                        
                        # Destacar próximo ciclo
                        next_cycle = next((p for p in selected_entity["revisit_points"] 
                                          if p["days_remaining"] > 0), None)
                        
                        if next_cycle:
                            st.markdown(f"""
                            **Próximo Ciclo:** {next_cycle['cycle']}  
                            **Data:** {next_cycle['date']}  
                            **Dias Restantes:** {next_cycle['days_remaining']}
                            """)
                    
                    # Visualizações
                    st.markdown("### Visualizações")
                    
                    tab_vis1, tab_vis2 = st.tabs(["Curva de Decaimento", "Timeline"])
                    
                    with tab_vis1:
                        fig1 = self.plot_decay_curve(selected_entity['id'])
                        st.pyplot(fig1)
                    
                    with tab_vis2:
                        fig2 = self.plot_entity_timeline(selected_entity['id'])
                        st.pyplot(fig2)
                    
                    # Opções avançadas
                    with st.expander("Opções Avançadas"):
                        col_opt1, col_opt2 = st.columns(2)
                        
                        with col_opt1:
                            new_state = st.selectbox(
                                "Atualizar Estado", 
                                self.decay_states,
                                index=self.decay_states.index(selected_entity['decay_state']),
                                key="update_state"
                            )
                            
                            if st.button("Atualizar Estado", key="btn_update_state"):
                                if self.update_entity(selected_entity['id'], 'decay_state', new_state):
                                    st.success("Estado atualizado com sucesso!")
                                    st.experimental_rerun()
                        
                        with col_opt2:
                            new_half_life = st.selectbox(
                                "Atualizar Half-Life", 
                                self.half_life_types,
                                index=self.half_life_types.index(selected_entity['half_life_type']),
                                key="update_halflife"
                            )
                            
                            if st.button("Atualizar Half-Life", key="btn_update_halflife"):
                                if self.update_entity(selected_entity['id'], 'half_life_type', new_half_life):
                                    st.success("Half-Life atualizado com sucesso!")
                                    st.experimental_rerun()
                        
                        # Opção para excluir
                        if st.button("🗑️ Excluir Entidade", key="btn_delete"):
                            if self.delete_entity(selected_entity['id']):
                                st.success("Entidade excluída com sucesso!")
                                st.experimental_rerun()
                            else:
                                st.error("Erro ao excluir entidade.")
        
        # Tab 3: Próximos Ciclos
        with tab3:
            st.markdown("### Próximos Ciclos de Reconsolidação")
            
            if not self.entities:
                st.info("Nenhuma entidade registrada ainda.")
            else:
                # Obter próximos ciclos
                days_ahead = st.slider("Dias à frente", 30, 365, 90, step=30, key="days_ahead")
                upcoming = self.get_next_revisits(days_ahead=days_ahead, max_items=20)
                
                if not upcoming:
                    st.info(f"Nenhum ciclo de reconsolidação nos próximos {days_ahead} dias.")
                else:
                    # Criar dataframe para visualização
                    upcoming_df = pd.DataFrame(upcoming)
                    upcoming_df = upcoming_df[["entity_name", "entity_category", "revisit_date", 
                                           "days_remaining", "cycle", "percentage"]]
                    upcoming_df.columns = ["Entidade", "Categoria", "Data", 
                                        "Dias Restantes", "Ciclo", "Intensidade (%)"]
                    
                    st.dataframe(upcoming_df, use_container_width=True)
                    
                    # Criar visualização de calendário
                    st.markdown("### Calendário de Reconsolidação")
                    
                    # Determinar intervalo de datas
                    start_date = self.today
                    end_date = self.today + datetime.timedelta(days=days_ahead)
                    
                    # Preparar dados para o gráfico
                    dates = []
                    counts = []
                    
                    current = start_date
                    while current <= end_date:
                        dates.append(current)
                        # Contar ciclos para esta data
                        count = sum(1 for r in upcoming 
                                   if datetime.datetime.strptime(r["revisit_date"], "%Y-%m-%d").date() == current)
                        counts.append(count)
                        current += datetime.timedelta(days=1)
                    
                    # Criar figura
                    fig, ax = plt.subplots(figsize=(12, 5))
                    fig.patch.set_facecolor('#0E1117')
                    ax.set_facecolor('#0E1117')
                    
                    # Configurar eixos
                    ax.spines['bottom'].set_color('#CCCCCC')
                    ax.spines['left'].set_color('#CCCCCC')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.tick_params(colors='#CCCCCC')
                    
                    # Plotar barras
                    bars = ax.bar(range(len(dates)), counts, color='#42A5F5', alpha=0.7)
                    
                    # Destacar barras com ciclos
                    for i, count in enumerate(counts):
                        if count > 0:
                            bars[i].set_color('#FF5252')
                            bars[i].set_alpha(0.9)
                    
                    # Configurar rótulos
                    ax.set_xlabel('Data', color='#FFFFFF', fontsize=12)
                    ax.set_ylabel('Número de Ciclos', color='#FFFFFF', fontsize=12)
                    ax.set_title('Calendário de Reconsolidação', color='#FFFFFF', fontsize=14)
                    
                    # Configurar ticks do eixo x (datas)
                    tick_positions = list(range(0, len(dates), max(1, len(dates) // 10)))
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels([dates[pos].strftime("%d/%m/%Y") for pos in tick_positions], 
                                     rotation=45, ha='right')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        # Tab 4: Documentação
        with tab4:
            st.markdown("### Documentação do Ritual de Decaimento Fractal")
            
            st.markdown("""
            Este módulo implementa o conceito de **Decaimento Preservando Memória de Campo**, conforme documentado em `FRACTAL_DECAY_REF_001.md`.
            
            #### Princípios Fundamentais:
            
            1. **Decaimento Rítmico:** O half-life demonstra que mesmo a desintegração segue padrões matemáticos precisos, permitindo previsibilidade em meio ao aparente caos.
            
            2. **Ressonância de Campo:** A memória do que existiu permanece impressa no campo, mesmo quando a forma original desaparece.
            
            3. **Reconsolidação Fractal:** Elementos que decaem seguem padrões auto-similares através do tempo, permitindo reconstrução parcial através da análise fractal.
            
            #### Como Utilizar:
            
            1. **Registro de Entidades:** Documente entidades (pessoas, animais, lugares, objetos, etc.) que estão em processo de transformação/decaimento, mas cuja memória você deseja preservar.
            
            2. **Definição de Half-Life:** Escolha um tipo de meia-vida que melhor representa o ritmo de decaimento memorial/emocional da entidade.
            
            3. **Cronograma de Reconsolidação:** O sistema calculará automaticamente os pontos ideais para revisitar e reconectar com a memória da entidade.
            
            4. **Artefatos de Ancoragem:** Registre objetos físicos ou práticas que servem como pontos de acesso à memória da entidade.
            
            #### Aplicações Práticas:
            
            - **Memorial:** Honrar e preservar memórias de entes queridos
            - **Temporal:** Acompanhar a transformação de projetos e conceitos ao longo do tempo
            - **Espacial:** Documentar lugares em transformação
            - **Relacional:** Mapear o decaimento e transformação de relações
            """)
            
            if st.button("Abrir Referência Conceitual", key="open_ref"):
                st.session_state.current_document = "TECNOLOGIAS/FRACTAL_DECAY_REF_001.md"
                
                # Garantir que este documento está na lista
                if "TECNOLOGIAS/FRACTAL_DECAY_REF_001.md" not in [d["name"] for d in st.session_state.document_list]:
                    st.session_state.document_list.append({
                        "name": "TECNOLOGIAS/FRACTAL_DECAY_REF_001.md",
                        "type": "markdown",
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                # Mudar para a aba de documentos
                st.experimental_rerun()

def show_interface():
    """Função para mostrar a interface no app Streamlit"""
    decay_ritual = FractalDecayRitual()
    decay_ritual.render_ui()

if __name__ == "__main__":
    show_interface()