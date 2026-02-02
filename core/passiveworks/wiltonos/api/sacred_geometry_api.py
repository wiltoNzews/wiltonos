"""
API de Geometria Sagrada para WiltonOS

Esta API fornece endpoints para gerar e acessar imagens de geometria sagrada.
As imagens são geradas usando algoritmos matemáticos que respeitam
as proporções e padrões sagrados.
"""

import os
import json
from flask import Blueprint, request, jsonify, send_from_directory

# Importar gerador de geometria sagrada
from ..sacred_geometry.geometry_generator import (
    flower_of_life, metatrons_cube, sri_yantra, merkaba, torus, generate_all_patterns
)

# Diretório para imagens de geometria sagrada
PUBLIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'public')
SACRED_GEOMETRY_DIR = os.path.join(PUBLIC_DIR, 'sacred_geometry')

# Criar o Blueprint
sacred_geometry_api = Blueprint('sacred_geometry_api', __name__)

@sacred_geometry_api.route('/api/sacred-geometry/generate', methods=['POST'])
def generate_sacred_geometry():
    """
    Gera uma imagem de geometria sagrada com os parâmetros especificados
    
    Parâmetros:
        type: Tipo de geometria ('flower_of_life', 'metatrons_cube', 'sri_yantra', 'merkaba', 'torus', 'all')
        size: Tamanho da imagem (opcional, padrão: 1000)
        color_mode: Modo de cor ('rainbow', 'golden', 'monochrome', 'quantum') (opcional, padrão: 'quantum')
    
    Retorna:
        Informações da(s) imagem(ns) gerada(s)
    """
    try:
        data = request.get_json() or {}
        
        # Validar parâmetros
        type_param = data.get('type', 'merkaba').lower()
        size = int(data.get('size', 1000))
        color_mode = data.get('color_mode', 'quantum').lower()
        
        # Limitar tamanho para evitar sobrecarga
        if size < 100:
            size = 100
        elif size > 2000:
            size = 2000
            
        # Verificar modo de cor válido
        valid_color_modes = ['rainbow', 'golden', 'monochrome', 'quantum']
        if color_mode not in valid_color_modes:
            color_mode = 'quantum'
            
        # Verificar tipo de geometria válido
        valid_types = ['flower_of_life', 'metatrons_cube', 'sri_yantra', 'merkaba', 'torus', 'all']
        if type_param not in valid_types:
            return jsonify({
                'success': False,
                'message': f'Tipo de geometria inválido. Use um dos seguintes: {", ".join(valid_types)}'
            }), 400
            
        # Gerar imagem com base no tipo selecionado
        if type_param == 'flower_of_life':
            result = flower_of_life(size=size, color_mode=color_mode)
            return jsonify(result)
            
        elif type_param == 'metatrons_cube':
            result = metatrons_cube(size=size, color_mode=color_mode)
            return jsonify(result)
            
        elif type_param == 'sri_yantra':
            result = sri_yantra(size=size, color_mode=color_mode)
            return jsonify(result)
            
        elif type_param == 'merkaba':
            result = merkaba(size=size, color_mode=color_mode)
            return jsonify(result)
            
        elif type_param == 'torus':
            result = torus(size=size, color_mode=color_mode)
            return jsonify(result)
            
        elif type_param == 'all':
            results = generate_all_patterns(size=size, color_mode=color_mode)
            return jsonify({
                'success': True,
                'message': f'Todas as geometrias geradas no modo {color_mode}',
                'images': results
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erro ao gerar geometria sagrada: {str(e)}'
        }), 500

@sacred_geometry_api.route('/sacred_geometry/<filename>', methods=['GET'])
def get_sacred_geometry_image(filename):
    """
    Retorna uma imagem de geometria sagrada pelo nome do arquivo
    """
    try:
        return send_from_directory(SACRED_GEOMETRY_DIR, filename)
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erro ao buscar imagem: {str(e)}'
        }), 404

@sacred_geometry_api.route('/api/sacred-geometry/list', methods=['GET'])
def list_sacred_geometry():
    """
    Lista todas as imagens de geometria sagrada disponíveis
    """
    try:
        if not os.path.exists(SACRED_GEOMETRY_DIR):
            return jsonify({
                'success': True,
                'message': 'Nenhuma imagem encontrada',
                'images': []
            })
            
        files = os.listdir(SACRED_GEOMETRY_DIR)
        images = []
        
        for file in files:
            if file.endswith('.png'):
                images.append({
                    'filename': file,
                    'url': f'/sacred_geometry/{file}'
                })
                
        return jsonify({
            'success': True,
            'message': f'{len(images)} imagens encontradas',
            'images': images
        })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erro ao listar imagens: {str(e)}'
        }), 500