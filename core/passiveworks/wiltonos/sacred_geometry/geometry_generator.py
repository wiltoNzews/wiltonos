"""
WiltonOS - Gerador de Geometria Sagrada

Este módulo gera imagens de geometria sagrada usando algoritmos matemáticos
e a biblioteca Pillow. Diferentes padrões são implementados como funções
independentes.
"""

import os
import math
import random
from PIL import Image, ImageDraw, ImageFilter, ImageColor, ImageOps, ImageChops

# Diretório para salvar as imagens geradas
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'public', 'sacred_geometry')

# Garantir que o diretório de saída exista
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def golden_ratio():
    """Retorna o número de ouro"""
    return (1 + math.sqrt(5)) / 2

def hsv_to_rgb(h, s, v):
    """Converte cores HSV para RGB"""
    if s == 0.0:
        return (v, v, v)
    
    i = int(h * 6)
    f = (h * 6) - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    
    i %= 6
    
    if i == 0:
        return (v, t, p)
    elif i == 1:
        return (q, v, p)
    elif i == 2:
        return (p, v, t)
    elif i == 3:
        return (p, q, v)
    elif i == 4:
        return (t, p, v)
    else:
        return (v, p, q)

def apply_quantum_noise(image, intensity=0.05):
    """Aplica ruído quântico à imagem para criar texturas"""
    pixels = image.load()
    width, height = image.size
    
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            
            # Ruído aleatório com distribuição quântica
            noise = random.uniform(-intensity, intensity)
            
            # Aplicar ruído baseado no número de ouro
            golden = golden_ratio()
            r = int(max(0, min(255, r + r * noise * golden)))
            g = int(max(0, min(255, g + g * noise * (1/golden))))
            b = int(max(0, min(255, b + b * noise)))
            
            pixels[x, y] = (r, g, b, a)
    
    return image

def flower_of_life(size=1000, background_color=(0, 0, 0, 255), circles=7, color_mode="rainbow"):
    """
    Gera uma imagem com o padrão Flor da Vida
    
    Args:
        size: Tamanho da imagem em pixels
        background_color: Cor de fundo (RGBA)
        circles: Número de círculos
        color_mode: "rainbow", "golden", "monochrome" ou "quantum"
        
    Returns:
        Caminho da imagem gerada
    """
    # Criar imagem base
    image = Image.new('RGBA', (size, size), background_color)
    draw = ImageDraw.Draw(image)
    
    # Tamanho e posição base
    radius = size / (circles * 2)
    center_x, center_y = size / 2, size / 2
    
    # Desenhar primeiro círculo central
    if color_mode == "rainbow":
        circle_color = (129, 50, 168, 128)
    elif color_mode == "golden":
        circle_color = (212, 175, 55, 128)
    elif color_mode == "quantum":
        circle_color = (109, 40, 217, 128)
    else:  # monochrome
        circle_color = (255, 255, 255, 128)
    
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        outline=circle_color, width=2
    )
    
    # Criar padrão de círculos
    for layer in range(1, circles):
        # Posições para esta camada
        num_positions = 6 * layer
        for pos in range(num_positions):
            angle = 2 * math.pi * pos / num_positions
            x = center_x + (layer * radius * math.cos(angle) * 2)
            y = center_y + (layer * radius * math.sin(angle) * 2)
            
            # Calcular cor baseada no modo
            if color_mode == "rainbow":
                hue = pos / num_positions
                r, g, b = hsv_to_rgb(hue, 0.7, 0.9)
                circle_color = (int(r*255), int(g*255), int(b*255), 128)
            elif color_mode == "golden":
                golden = golden_ratio()
                ratio = (pos / num_positions) * golden
                circle_color = (
                    int(212 * ratio), 
                    int(175 * ratio), 
                    int(55 * ratio), 
                    128
                )
            elif color_mode == "quantum":
                # Roxa a azul com variação quântica
                ratio = (pos / num_positions)
                circle_color = (
                    int(109 * (1-ratio)), 
                    int(40 * ratio), 
                    int(217 * ratio), 
                    128
                )
            
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=circle_color, width=2
            )
    
    # Aplicar efeito de desfoque suave
    image = image.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Aplicar efeito de ruído quântico
    if color_mode == "quantum":
        image = apply_quantum_noise(image, 0.08)
    
    # Salvar imagem
    filename = f"flower_of_life_{color_mode}_{size}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    image.save(output_path)
    
    return {
        "success": True,
        "message": f"Flor da Vida gerada no modo {color_mode}",
        "path": output_path,
        "filename": filename,
        "url": f"/sacred_geometry/{filename}"
    }

def metatrons_cube(size=1000, background_color=(0, 0, 0, 255), color_mode="rainbow"):
    """
    Gera uma imagem com o Cubo de Metatron
    
    Args:
        size: Tamanho da imagem em pixels
        background_color: Cor de fundo (RGBA)
        color_mode: "rainbow", "golden", "monochrome" ou "quantum"
        
    Returns:
        Caminho da imagem gerada
    """
    # Criar imagem base
    image = Image.new('RGBA', (size, size), background_color)
    draw = ImageDraw.Draw(image)
    
    # Tamanho e posição base
    radius = size / 4
    center_x, center_y = size / 2, size / 2
    
    # Pontos do Cubo de Metatron (baseados em um dodecaedro)
    points = []
    
    # Círculo central
    draw.ellipse((center_x - radius/2, center_y - radius/2, 
                  center_x + radius/2, center_y + radius/2),
                 outline=(255, 255, 255, 180), width=2)
    
    # Adicionar 12 pontos do dodecaedro
    golden = golden_ratio()
    
    # Primeiros 4 pontos (plano xy)
    points.append((center_x + radius, center_y))
    points.append((center_x, center_y + radius))
    points.append((center_x - radius, center_y))
    points.append((center_x, center_y - radius))
    
    # Próximos 4 pontos (plano yz escalado pelo número de ouro)
    points.append((center_x + radius/golden, center_y + radius/golden))
    points.append((center_x - radius/golden, center_y + radius/golden))
    points.append((center_x - radius/golden, center_y - radius/golden))
    points.append((center_x + radius/golden, center_y - radius/golden))
    
    # Últimos 4 pontos (plano xz escalado pelo número de ouro)
    r2 = radius * (1 + 1/golden)/2
    points.append((center_x + r2, center_y + r2/golden))
    points.append((center_x - r2, center_y + r2/golden))
    points.append((center_x - r2, center_y - r2/golden))
    points.append((center_x + r2, center_y - r2/golden))
    
    # Desenhar círculos em cada ponto
    for i, (x, y) in enumerate(points):
        if color_mode == "rainbow":
            hue = i / len(points)
            r, g, b = hsv_to_rgb(hue, 0.8, 0.9)
            circle_color = (int(r*255), int(g*255), int(b*255), 180)
        elif color_mode == "golden":
            ratio = (i / len(points)) * golden
            circle_color = (
                int(212 * ratio), 
                int(175 * ratio), 
                int(55 * ratio), 
                180
            )
        elif color_mode == "quantum":
            ratio = (i / len(points))
            circle_color = (
                int(109 * (1-ratio)), 
                int(40 * ratio), 
                int(217 * ratio), 
                180
            )
        else:  # monochrome
            circle_color = (255, 255, 255, 180)
            
        draw.ellipse((x - radius/4, y - radius/4, x + radius/4, y + radius/4),
                     outline=circle_color, width=2)
    
    # Desenhar linhas conectando todos os pontos
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points[i+1:], i+1):
            if color_mode == "rainbow":
                hue = (i + j) / (len(points) * 2)
                r, g, b = hsv_to_rgb(hue, 0.7, 0.9)
                line_color = (int(r*255), int(g*255), int(b*255), 100)
            elif color_mode == "golden":
                ratio = ((i + j) / (len(points) * 2)) * golden
                line_color = (
                    int(212 * ratio), 
                    int(175 * ratio), 
                    int(55 * ratio), 
                    100
                )
            elif color_mode == "quantum":
                ratio = (i + j) / (len(points) * 2)
                line_color = (
                    int(109 * (1-ratio)), 
                    int(40 * ratio), 
                    int(217 * ratio), 
                    100
                )
            else:  # monochrome
                line_color = (255, 255, 255, 100)
                
            draw.line([p1, p2], fill=line_color, width=1)
    
    # Aplicar efeito de desfoque suave
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Aplicar efeito de ruído quântico
    if color_mode == "quantum":
        image = apply_quantum_noise(image, 0.08)
    
    # Salvar imagem
    filename = f"metatrons_cube_{color_mode}_{size}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    image.save(output_path)
    
    return {
        "success": True,
        "message": f"Cubo de Metatron gerado no modo {color_mode}",
        "path": output_path,
        "filename": filename,
        "url": f"/sacred_geometry/{filename}"
    }

def sri_yantra(size=1000, background_color=(0, 0, 0, 255), color_mode="rainbow"):
    """
    Gera uma imagem do Sri Yantra
    
    Args:
        size: Tamanho da imagem em pixels
        background_color: Cor de fundo (RGBA)
        color_mode: "rainbow", "golden", "monochrome" ou "quantum"
        
    Returns:
        Caminho da imagem gerada
    """
    # Criar imagem base
    image = Image.new('RGBA', (size, size), background_color)
    draw = ImageDraw.Draw(image)
    
    # Tamanho e posição base
    radius = size / 2.5
    center_x, center_y = size / 2, size / 2
    
    # Desenhar círculo externo
    if color_mode == "rainbow":
        circle_color = (129, 50, 168, 128)
    elif color_mode == "golden":
        circle_color = (212, 175, 55, 128)
    elif color_mode == "quantum":
        circle_color = (109, 40, 217, 128)
    else:  # monochrome
        circle_color = (255, 255, 255, 128)
        
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        outline=circle_color, width=2
    )
    
    # Desenhar os triângulos do Sri Yantra
    # Nove triângulos interligados: 4 apontando para cima, 5 apontando para baixo
    triangles = []
    
    # Triângulos apontando para cima
    for i in range(4):
        ratio = 0.5 + (i * 0.1)
        r = radius * ratio
        
        # Calcular pontos do triângulo
        p1 = (center_x, center_y - r)
        p2 = (center_x - r * math.sin(math.radians(60)), center_y + r * math.cos(math.radians(60)))
        p3 = (center_x + r * math.sin(math.radians(60)), center_y + r * math.cos(math.radians(60)))
        
        triangles.append((p1, p2, p3))
    
    # Triângulos apontando para baixo
    for i in range(5):
        ratio = 0.45 + (i * 0.1)
        r = radius * ratio
        
        # Calcular pontos do triângulo
        p1 = (center_x, center_y + r)
        p2 = (center_x - r * math.sin(math.radians(60)), center_y - r * math.cos(math.radians(60)))
        p3 = (center_x + r * math.sin(math.radians(60)), center_y - r * math.cos(math.radians(60)))
        
        triangles.append((p1, p2, p3))
    
    # Desenhar os triângulos
    for i, triangle in enumerate(triangles):
        if color_mode == "rainbow":
            hue = i / len(triangles)
            r, g, b = hsv_to_rgb(hue, 0.8, 0.9)
            tri_color = (int(r*255), int(g*255), int(b*255), 128)
        elif color_mode == "golden":
            ratio = (i / len(triangles)) * golden_ratio()
            tri_color = (
                int(212 * ratio), 
                int(175 * ratio), 
                int(55 * ratio), 
                128
            )
        elif color_mode == "quantum":
            ratio = i / len(triangles)
            tri_color = (
                int(109 * (1-ratio)), 
                int(40 * ratio), 
                int(217 * ratio), 
                128
            )
        else:  # monochrome
            tri_color = (255, 255, 255, 128)
            
        draw.polygon(triangle, outline=tri_color, width=2)
    
    # Desenhar o ponto central (bindu)
    draw.ellipse(
        (center_x - radius/15, center_y - radius/15, center_x + radius/15, center_y + radius/15),
        fill=(255, 255, 255, 200)
    )
    
    # Desenhar os lótus de 8 pétalas
    num_petals = 8
    petal_radius = radius * 0.2
    for i in range(num_petals):
        angle = 2 * math.pi * i / num_petals
        x1 = center_x + (radius * 0.7) * math.cos(angle)
        y1 = center_y + (radius * 0.7) * math.sin(angle)
        
        # Calcular pontos da pétala
        petal_angle1 = angle - math.pi / 16
        petal_angle2 = angle + math.pi / 16
        
        x2 = center_x + (radius * 0.8) * math.cos(petal_angle1)
        y2 = center_y + (radius * 0.8) * math.sin(petal_angle1)
        x3 = center_x + (radius * 0.8) * math.cos(petal_angle2)
        y3 = center_y + (radius * 0.8) * math.sin(petal_angle2)
        
        if color_mode == "rainbow":
            hue = i / num_petals
            r, g, b = hsv_to_rgb(hue, 0.7, 0.9)
            petal_color = (int(r*255), int(g*255), int(b*255), 150)
        elif color_mode == "golden":
            ratio = (i / num_petals) * golden_ratio()
            petal_color = (
                int(212 * ratio), 
                int(175 * ratio), 
                int(55 * ratio), 
                150
            )
        elif color_mode == "quantum":
            ratio = i / num_petals
            petal_color = (
                int(109 * (1-ratio)), 
                int(40 * ratio), 
                int(217 * ratio), 
                150
            )
        else:  # monochrome
            petal_color = (255, 255, 255, 150)
            
        draw.polygon([(center_x, center_y), (x2, y2), (x3, y3)], outline=petal_color, width=2)
    
    # Aplicar efeito de desfoque suave
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Aplicar efeito de ruído quântico
    if color_mode == "quantum":
        image = apply_quantum_noise(image, 0.07)
    
    # Salvar imagem
    filename = f"sri_yantra_{color_mode}_{size}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    image.save(output_path)
    
    return {
        "success": True,
        "message": f"Sri Yantra gerado no modo {color_mode}",
        "path": output_path,
        "filename": filename,
        "url": f"/sacred_geometry/{filename}"
    }

def merkaba(size=1000, background_color=(0, 0, 0, 255), color_mode="rainbow"):
    """
    Gera uma imagem do Merkaba (Estrela Tetraédrica)
    
    Args:
        size: Tamanho da imagem em pixels
        background_color: Cor de fundo (RGBA)
        color_mode: "rainbow", "golden", "monochrome" ou "quantum"
        
    Returns:
        Caminho da imagem gerada
    """
    # Criar imagem base
    image = Image.new('RGBA', (size, size), background_color)
    draw = ImageDraw.Draw(image)
    
    # Tamanho e posição base
    radius = size / 3
    center_x, center_y = size / 2, size / 2
    
    # Calcular pontos do tetraedro (triângulo 3D projetado)
    # Primeiro tetraedro (apontando para cima)
    up_apex = (center_x, center_y - radius)
    up_base1 = (center_x - radius * math.sin(math.radians(60)), center_y + radius * math.cos(math.radians(60)))
    up_base2 = (center_x + radius * math.sin(math.radians(60)), center_y + radius * math.cos(math.radians(60)))
    up_base3 = (center_x, center_y + radius / 3)  # Ponto central da base (projeção)
    
    # Segundo tetraedro (apontando para baixo)
    down_apex = (center_x, center_y + radius)
    down_base1 = (center_x - radius * math.sin(math.radians(60)), center_y - radius * math.cos(math.radians(60)))
    down_base2 = (center_x + radius * math.sin(math.radians(60)), center_y - radius * math.cos(math.radians(60)))
    down_base3 = (center_x, center_y - radius / 3)  # Ponto central da base (projeção)
    
    # Definir cores com base no modo
    if color_mode == "rainbow":
        up_color = (50, 190, 255, 150)  # Azul
        down_color = (255, 50, 190, 150)  # Rosa
    elif color_mode == "golden":
        up_color = (212, 175, 55, 150)  # Dourado
        down_color = (255, 215, 0, 150)  # Ouro
    elif color_mode == "quantum":
        up_color = (109, 40, 217, 150)  # Roxo
        down_color = (40, 109, 217, 150)  # Azul
    else:  # monochrome
        up_color = (255, 255, 255, 150)
        down_color = (200, 200, 200, 150)
    
    # Desenhar linhas do primeiro tetraedro
    draw.line([up_apex, up_base1], fill=up_color, width=2)
    draw.line([up_apex, up_base2], fill=up_color, width=2)
    draw.line([up_apex, up_base3], fill=up_color, width=2)
    draw.line([up_base1, up_base2], fill=up_color, width=2)
    draw.line([up_base1, up_base3], fill=up_color, width=2)
    draw.line([up_base2, up_base3], fill=up_color, width=2)
    
    # Desenhar linhas do segundo tetraedro
    draw.line([down_apex, down_base1], fill=down_color, width=2)
    draw.line([down_apex, down_base2], fill=down_color, width=2)
    draw.line([down_apex, down_base3], fill=down_color, width=2)
    draw.line([down_base1, down_base2], fill=down_color, width=2)
    draw.line([down_base1, down_base3], fill=down_color, width=2)
    draw.line([down_base2, down_base3], fill=down_color, width=2)
    
    # Desenhar círculo central
    center_color = (255, 255, 255, 100)
    draw.ellipse(
        (center_x - radius/8, center_y - radius/8, center_x + radius/8, center_y + radius/8),
        outline=center_color, width=2
    )
    
    # Adicionar linhas adicionais para criar a impressão 3D
    for i in range(1, 6):
        # Variar tamanho para dar profundidade
        r = radius * (0.8 + i * 0.04)
        
        # Calcular pontos
        p1 = (center_x, center_y - r/2)
        p2 = (center_x - r/2, center_y + r/3)
        p3 = (center_x + r/2, center_y + r/3)
        
        # Definir cor
        if color_mode == "rainbow":
            hue = i / 6
            r_val, g, b = hsv_to_rgb(hue, 0.7, 0.9)
            line_color = (int(r_val*255), int(g*255), int(b*255), 80)
        elif color_mode == "golden":
            ratio = (i / 6) * golden_ratio()
            line_color = (int(212 * ratio), int(175 * ratio), int(55 * ratio), 80)
        elif color_mode == "quantum":
            ratio = i / 6
            line_color = (int(109 * (1-ratio)), int(40 * ratio), int(217 * ratio), 80)
        else:  # monochrome
            line_color = (255, 255, 255, 80)
            
        # Desenhar linhas
        draw.line([p1, p2], fill=line_color, width=1)
        draw.line([p1, p3], fill=line_color, width=1)
        draw.line([p2, p3], fill=line_color, width=1)
    
    # Aplicar efeito de desfoque suave
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Aplicar efeito de ruído quântico
    if color_mode == "quantum":
        image = apply_quantum_noise(image, 0.06)
    
    # Salvar imagem
    filename = f"merkaba_{color_mode}_{size}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    image.save(output_path)
    
    return {
        "success": True,
        "message": f"Merkaba gerado no modo {color_mode}",
        "path": output_path,
        "filename": filename,
        "url": f"/sacred_geometry/{filename}"
    }

def torus(size=1000, background_color=(0, 0, 0, 255), color_mode="rainbow"):
    """
    Gera uma imagem do Torus (visualização 2D)
    
    Args:
        size: Tamanho da imagem em pixels
        background_color: Cor de fundo (RGBA)
        color_mode: "rainbow", "golden", "monochrome" ou "quantum"
        
    Returns:
        Caminho da imagem gerada
    """
    # Criar imagem base
    image = Image.new('RGBA', (size, size), background_color)
    draw = ImageDraw.Draw(image)
    
    # Tamanho e posição base
    radius_outer = size / 3
    radius_inner = radius_outer / 2
    center_x, center_y = size / 2, size / 2
    
    # Desenhar círculo externo
    if color_mode == "rainbow":
        circle_color_outer = (128, 0, 128, 180)  # Purple
    elif color_mode == "golden":
        circle_color_outer = (212, 175, 55, 180)  # Gold
    elif color_mode == "quantum":
        circle_color_outer = (109, 40, 217, 180)  # Purple-blue
    else:  # monochrome
        circle_color_outer = (255, 255, 255, 180)
        
    draw.ellipse(
        (center_x - radius_outer, center_y - radius_outer*0.4, 
         center_x + radius_outer, center_y + radius_outer*0.4),
        outline=circle_color_outer, width=2
    )
    
    # Desenhar círculo interno
    if color_mode == "rainbow":
        circle_color_inner = (0, 128, 255, 180)  # Blue
    elif color_mode == "golden":
        circle_color_inner = (255, 215, 0, 180)  # Yellow gold
    elif color_mode == "quantum":
        circle_color_inner = (40, 109, 217, 180)  # Blue
    else:  # monochrome
        circle_color_inner = (220, 220, 220, 180)
        
    draw.ellipse(
        (center_x - radius_inner, center_y - radius_inner*0.4,
         center_x + radius_inner, center_y + radius_inner*0.4),
        outline=circle_color_inner, width=2
    )
    
    # Desenhar linhas representando fluxos de energia
    num_lines = 36
    for i in range(num_lines):
        angle = 2 * math.pi * i / num_lines
        
        # Calcular pontos para linhas curvas
        points = []
        for t in range(0, 101, 5):
            t_rad = t / 100 * 2 * math.pi
            
            # Equações paramétricas da curva do torus
            r = radius_outer*0.7 + radius_inner*0.7 * math.cos(t_rad + angle)
            x = center_x + r * math.cos(angle * 0.1 + t_rad/3)
            y = center_y + r * 0.4 * math.sin(angle * 0.1 + t_rad/3)
            
            points.append((x, y))
        
        # Definir cor com base no modo
        if color_mode == "rainbow":
            hue = i / num_lines
            r, g, b = hsv_to_rgb(hue, 0.7, 0.9)
            line_color = (int(r*255), int(g*255), int(b*255), 100)
        elif color_mode == "golden":
            ratio = (i / num_lines) * golden_ratio()
            line_color = (
                int(212 * ratio), 
                int(175 * ratio), 
                int(55 * ratio), 
                100
            )
        elif color_mode == "quantum":
            ratio = i / num_lines
            line_color = (
                int(109 * (1-ratio)), 
                int(40 * ratio), 
                int(217 * ratio), 
                100
            )
        else:  # monochrome
            line_color = (255, 255, 255, 100)
            
        # Desenhar curva
        for j in range(len(points)-1):
            draw.line([points[j], points[j+1]], fill=line_color, width=1)
    
    # Aplicar efeito de desfoque suave
    image = image.filter(ImageFilter.GaussianBlur(radius=0.7))
    
    # Aplicar efeito de ruído quântico
    if color_mode == "quantum":
        image = apply_quantum_noise(image, 0.05)
    
    # Salvar imagem
    filename = f"torus_{color_mode}_{size}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    image.save(output_path)
    
    return {
        "success": True,
        "message": f"Torus gerado no modo {color_mode}",
        "path": output_path,
        "filename": filename,
        "url": f"/sacred_geometry/{filename}"
    }

def generate_all_patterns(size=1000, color_mode="quantum"):
    """
    Gera todas as imagens de geometria sagrada
    
    Args:
        size: Tamanho da imagem em pixels
        color_mode: "rainbow", "golden", "monochrome" ou "quantum"
        
    Returns:
        Lista com informações de todas as imagens geradas
    """
    results = []
    
    # Gerar Flor da Vida
    results.append(flower_of_life(size, color_mode=color_mode))
    
    # Gerar Cubo de Metatron
    results.append(metatrons_cube(size, color_mode=color_mode))
    
    # Gerar Sri Yantra
    results.append(sri_yantra(size, color_mode=color_mode))
    
    # Gerar Merkaba
    results.append(merkaba(size, color_mode=color_mode))
    
    # Gerar Torus
    results.append(torus(size, color_mode=color_mode))
    
    return results

if __name__ == "__main__":
    # Teste das funções
    flower_of_life(color_mode="quantum")
    metatrons_cube(color_mode="quantum")
    sri_yantra(color_mode="quantum")
    merkaba(color_mode="quantum")
    torus(color_mode="quantum")
    print("Imagens geradas com sucesso!")