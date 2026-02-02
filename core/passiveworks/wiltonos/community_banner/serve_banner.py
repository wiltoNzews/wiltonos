from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import sys

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

class CustomHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/output/free_as_fck_banner_5x2.png'
        return SimpleHTTPRequestHandler.do_GET(self)

def run(server_class=HTTPServer, handler_class=CustomHandler):
    server_address = ('', 8000)
    httpd = server_class(server_address, handler_class)
    print('Servidor iniciado em http://localhost:8000')
    print('Acesse a imagem diretamente em: http://localhost:8000/output/free_as_fck_banner_5x2.png')
    httpd.serve_forever()

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    run()