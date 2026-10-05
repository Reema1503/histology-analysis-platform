"""Run: python app.py. Open the URL printed in the terminal."""
import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from analysis import count_candidates

ROOT = Path(__file__).parent
MAX_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 16_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def decode_image(encoded):
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > MAX_BYTES:
        raise ValueError('Maximum file size is 25 MB.')
    im = Image.open(io.BytesIO(raw))
    if im.width * im.height > MAX_PIXELS:
        raise ValueError('Maximum image size is 16 megapixels. Export a smaller field.')
    if getattr(im, 'n_frames', 1) != 1:
        raise ValueError('Stacks and multi-page TIFFs are not supported. Export one RGB field.')
    if im.mode not in ('RGB', 'RGBA', 'L', 'P'):
        raise ValueError('Use an 8-bit RGB image. Raw 16-bit or separate-channel TIFF needs a dedicated importer.')
    return im.convert('RGB')


class Handler(BaseHTTPRequestHandler):
    def send(self, status, data, kind='application/json'):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.send(200, (ROOT/'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/health':
            self.send(200, {'status': 'ok'})
        else:
            self.send(404, {'error': 'Not found'})

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 36 * 1024 * 1024:
                raise ValueError('Request too large or empty.')
            request = json.loads(self.rfile.read(length))
            im = decode_image(request['image'])
            if self.path == '/api/image':
                buf = io.BytesIO()
                im.save(buf, format='PNG')
                self.send(200, {'png': base64.b64encode(buf.getvalue()).decode(), 'width': im.width, 'height': im.height})
            elif self.path == '/api/count':
                self.send(200, count_candidates(np.array(im), request.get('options', {})))
            else:
                self.send(404, {'error': 'Not found'})
        except (ValueError, KeyError, UnidentifiedImageError, Image.DecompressionBombError) as e:
            self.send(400, {'error': str(e)})
        except Exception:
            self.send(500, {'error': 'Could not process this image. Check format and settings.'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8501'))
    host = os.environ.get('HOST', '127.0.0.1')
    print(f'Histology Workbench: http://{host}:{port}', flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
