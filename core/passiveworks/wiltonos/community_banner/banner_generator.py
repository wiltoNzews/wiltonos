import os
import sys
import math
import random
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageColor

# Diretório de saída para o banner
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')

# Cores disponíveis
COLOR_SCHEMES = {
    'purple': {
        'bg_start': (20, 10, 40),
        'bg_end': (40, 20, 80),
        'primary': (109, 40, 217),
        'secondary': (139, 92, 246),
        'accent': (243, 232, 255)
    },
    'gold': {
        'bg_start': (30, 20, 10),
        'bg_end': (50, 40, 20),
        'primary': (217, 175, 40),
        'secondary': (246, 213, 92),
        'accent': (255, 248, 232)
    },
    'red': {
        'bg_start': (40, 10, 10),
        'bg_end': (80, 20, 20),
        'primary': (220, 38, 38),
        'secondary': (248, 113, 113),
        'accent': (254, 242, 242)
    },
    'blue': {
        'bg_start': (10, 20, 40),
        'bg_end': (20, 40, 80),
        'primary': (37, 99, 235),
        'secondary': (96, 165, 250),
        'accent': (219, 234, 254)
    }
}

# Assegurar que o diretório de saída exista
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def add_noise(image, intensity=0.05):
    """Adiciona ruído sutil à imagem para textura"""
    pixels = image.load()
    width, height = image.size
    
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            noise = random.uniform(-intensity, intensity)
            
            pixels[x, y] = (
                min(max(int(r + r * noise), 0), 255),
                min(max(int(g + g * noise), 0), 255),
                min(max(int(b + b * noise), 0), 255),
                a
            )
    
    return image

def draw_flower_of_life(draw, center_x, center_y, radius, color, num_circles=7):
    """Desenha a Flor da Vida - símbolo de geometria sagrada"""
    # Círculo central
    draw.ellipse((center_x - radius, center_y - radius, 
                  center_x + radius, center_y + radius), 
                 outline=color + (100,), width=2)
    
    if num_circles <= 1:
        return
    
    # Círculos externos em padrão hexagonal
    for i in range(6):
        angle = math.pi / 3 * i
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        
        draw.ellipse((x - radius, y - radius, 
                      x + radius, y + radius), 
                     outline=color + (100,), width=2)
        
        # Adicionar círculos de segunda camada se solicitado
        if num_circles > 6:
            for j in range(6):
                angle2 = math.pi / 3 * j
                x2 = x + radius * math.cos(angle2)
                y2 = y + radius * math.sin(angle2)
                
                draw.ellipse((x2 - radius, y2 - radius, 
                              x2 + radius, y2 + radius), 
                             outline=color + (80,), width=2)

def draw_quantum_symbol(draw, x, y, size, color):
    """Desenha um símbolo quântico estilizado"""
    # Base para o símbolo (probabilidade)
    draw.ellipse((x - size, y - size, x + size, y + size), 
                 outline=color + (160,), width=2)
    
    # Onda de superposição
    points = []
    for i in range(12):
        angle = math.pi * 2 * i / 12
        radius = size * (0.8 + 0.2 * math.sin(i * 5))
        points.append((x + radius * math.cos(angle), 
                       y + radius * math.sin(angle)))
    
    # Desenhar a onda
    if len(points) > 2:
        draw.line(points + [points[0]], fill=color + (200,), width=2)
    
    # Partícula central
    core_size = int(size * 0.2)
    draw.ellipse((x - core_size, y - core_size, 
                  x + core_size, y + core_size), 
                 fill=color + (230,))

def create_radial_gradient(width, height, center_color, edge_color):
    """Cria um gradiente radial para o fundo"""
    # Criar imagem base transparente
    base = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Calcular o raio para o gradiente
    radius = math.sqrt(width**2 + height**2) / 2
    
    for y in range(height):
        for x in range(width):
            # Distância do centro
            distance = math.sqrt((x - width/2)**2 + (y - height/2)**2)
            
            # Normalizar a distância
            ratio = min(1.0, distance / radius)
            
            # Interpolar cores
            r = int(center_color[0] * (1 - ratio) + edge_color[0] * ratio)
            g = int(center_color[1] * (1 - ratio) + edge_color[1] * ratio)
            b = int(center_color[2] * (1 - ratio) + edge_color[2] * ratio)
            
            # Definir pixel
            base.putpixel((x, y), (r, g, b, 255))
    
    return base

def generate_banner(text="FREE AS F*CK", color_scheme="purple", ratio=(5, 2), size=(1500, 600)):
    """Gera um banner da comunidade com geometria sagrada e símbolos quânticos"""
    # Obter o esquema de cores
    if color_scheme not in COLOR_SCHEMES:
        color_scheme = "purple"
    
    colors = COLOR_SCHEMES[color_scheme]
    
    # Criar imagem base com fundo gradiente
    img = create_radial_gradient(size[0], size[1], colors['bg_start'], colors['bg_end'])
    draw = ImageDraw.Draw(img)
    
    # Adicionar elementos de geometria sagrada
    # Flor da Vida na lateral esquerda
    draw_flower_of_life(draw, size[0] * 0.15, size[1] * 0.5, 
                        size[1] * 0.3, colors['primary'])
    
    # Símbolos quânticos distribuídos
    num_symbols = 12
    for _ in range(num_symbols):
        x = random.randint(int(size[0] * 0.3), int(size[0] * 0.9))
        y = random.randint(int(size[1] * 0.1), int(size[1] * 0.9))
        symbol_size = random.randint(int(size[1] * 0.02), int(size[1] * 0.06))
        
        # Alternar entre cores primária e secundária
        color = colors['primary'] if random.random() > 0.5 else colors['secondary']
        
        draw_quantum_symbol(draw, x, y, symbol_size, color)
    
    # Adicionar texto principal
    try:
        # Tenta carregar uma fonte moderna
        font_file = os.path.join(os.path.dirname(__file__), 'fonts', 'InstrumentSans-Medium.ttf')
        if os.path.exists(font_file):
            font = ImageFont.truetype(font_file, int(size[1] * 0.25))
        else:
            # Fallback para fonte padrão
            font = ImageFont.load_default()
            font = ImageFont.truetype(ImageFont.load_default().path, int(size[1] * 0.25))
    except Exception as e:
        print(f"Erro ao carregar fonte: {e}")
        # Fallback absoluto para uma fonte básica
        if hasattr(ImageFont, 'load_default'):
            font = ImageFont.load_default()
        else:
            font = ImageFont.truetype("arial.ttf", int(size[1] * 0.25))
    
    # Posicionar o texto
    text_color = tuple(colors['accent']) + (255,)
    text_position = (size[0] * 0.55, size[1] * 0.5)
    
    # Sombra sutil para o texto
    shadow_offset = int(size[1] * 0.01)
    shadow_color = (0, 0, 0, 150)
    draw.text((text_position[0] + shadow_offset, text_position[1] + shadow_offset),
              text, font=font, fill=shadow_color, anchor="mm")
    
    # Texto principal
    draw.text(text_position, text, font=font, fill=text_color, anchor="mm")
    
    # Adicionar borda sutil
    border_width = int(size[1] * 0.005)
    draw.rectangle([(0, 0), (size[0] - 1, size[1] - 1)], 
                   outline=tuple(colors['primary']) + (100,), width=border_width)
    
    # Adicionar ruído para textura
    img = add_noise(img, 0.03)
    
    # Aplicar um leve desfoque
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Salvar o banner
    output_path = os.path.join(OUTPUT_DIR, f"free_as_fck_banner_{ratio[0]}x{ratio[1]}.png")
    img.save(output_path)
    
    print(f"Banner gerado com sucesso: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerador de Banner da Comunidade")
    parser.add_argument("--text", type=str, default="FREE AS F*CK",
                        help="Texto principal do banner")
    parser.add_argument("--color-scheme", type=str, default="purple",
                        choices=list(COLOR_SCHEMES.keys()),
                        help="Esquema de cores do banner")
    parser.add_argument("--width", type=int, default=1500,
                        help="Largura do banner em pixels")
    parser.add_argument("--height", type=int, default=600,
                        help="Altura do banner em pixels")
    
    args = parser.parse_args()
    
    # Determinar a proporção a partir das dimensões
    gcd = math.gcd(args.width, args.height)
    ratio = (args.width // gcd, args.height // gcd)
    
    generate_banner(args.text, args.color_scheme, ratio, (args.width, args.height))