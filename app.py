"""Local server: python app.py. Original channel arrays stay in server memory."""
import base64
import io
import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from analysis import count_candidates, evaluate_points

ROOT = Path(__file__).parent
MAX_BYTES = 100 * 1024 * 1024
MAX_PIXELS = 16_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
IMAGES = {}  # Limit to 3 active fields; restart server to erase them.
LOCK = threading.Lock()


def png(a):
    buf = io.BytesIO(); Image.fromarray(np.asarray(a, np.uint8)).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def display(a):
    out = np.zeros(a.shape, np.uint8)
    for c in range(3):
        channel = a[:, :, c].astype(float)
        hi = max(float(np.percentile(channel, 99.9)), 1e-12)
        out[:, :, c] = np.clip(channel/hi*255, 0, 255).astype(np.uint8)
    return out


def decode_image(encoded, green_channel=1, blue_channel=2):
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > MAX_BYTES: raise ValueError('Maximum image file size is 100 MB.')
    axes = None
    source_shape = None
    if raw[:4] in (b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+'):
        with tifffile.TiffFile(io.BytesIO(raw)) as tf:
            if len(tf.series) != 1: raise ValueError('Multiple TIFF series unsupported. Export one field.')
            s = tf.series[0]; axes = s.axes; shape = s.shape; source_shape = list(shape)
            if len(shape) != 3 or not (axes in ('YXS', 'YXC', 'CYX', 'SYX')):
                raise ValueError(f'TIFF axes {axes}, shape {shape}: select/export one 2D multichannel field; Z/T stacks unsupported.')
            y, x = shape[axes.index('Y')], shape[axes.index('X')]
            if y*x > MAX_PIXELS: raise ValueError('Maximum field is 16 megapixels.')
            if max(shape) > MAX_PIXELS or np.prod(shape)*np.dtype(s.dtype).itemsize > 256*1024*1024:
                raise ValueError('Decoded TIFF too large.')
            a = s.asarray()
            if axes[0] in ('C', 'S'): a = np.moveaxis(a, 0, -1)
            fmt = 'TIFF'
    else:
        im = Image.open(io.BytesIO(raw)); fmt = im.format
        if getattr(im, 'n_frames', 1) != 1: raise ValueError('Animated or multipage images unsupported.')
        if im.width*im.height > MAX_PIXELS: raise ValueError('Maximum field is 16 megapixels.')
        if im.mode not in ('RGB', 'RGBA'): raise ValueError('Use RGB/RGBA or a single multichannel TIFF. Grayscale needs paired channels.')
        a = np.asarray(im); axes = 'YXS'
    if a.dtype.kind not in 'uif' or a.ndim != 3: raise ValueError('Unsupported channel data type.')
    gi, bi = int(green_channel), int(blue_channel)
    if gi == bi or min(gi, bi) < 0 or max(gi, bi) >= a.shape[2]:
        raise ValueError(f'Choose two different channel indices from 0 to {a.shape[2]-1}. RGB normally uses green=1, blue=2.')
    if not np.all(np.isfinite(a)) or np.any(a < 0): raise ValueError('Nonfinite or negative image values unsupported.')
    rgb = np.zeros((*a.shape[:2], 3), dtype=a.dtype)
    rgb[:, :, 1] = a[:, :, gi]; rgb[:, :, 2] = a[:, :, bi]
    if gi != 0 and bi != 0: rgb[:, :, 0] = a[:, :, 0]
    meta = dict(format=fmt, width=a.shape[1], height=a.shape[0], axes=axes,
                source_shape=source_shape or list(a.shape), channels=a.shape[2], dtype=str(a.dtype),
                storage_bits=a.dtype.itemsize*8, green_channel=gi, blue_channel=bi,
                channel_ranges=[[float(a[:, :, i].min()), float(a[:, :, i].max())] for i in range(a.shape[2])],
                calibration='Not inferred; enter a verified pixel size if available.',
                quality_warning='Compressed/exported colour image; demonstration only.' if fmt in ('WEBP', 'JPEG') else 'Confirm original acquisition and channel mapping.')
    return rgb, meta


def annotated_png(a, review):
    """Full-resolution display export. Never changes the original channel array."""
    original = Image.fromarray(display(a))
    image = Image.new('RGB', (original.width, original.height+40), 'black')
    image.paste(original, (0, 0)); draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=12)
    h, w = a.shape[:2]; cells = review.get('cells', [])
    if len(cells) > 100000: raise ValueError('Too many markers for export.')
    roi = review.get('settings') or dict(x0=0, y0=0, x1=w, y1=h)
    final = 0
    for c in cells:
        x, y = float(c['x']), float(c['y'])
        if not np.isfinite(x+y) or not 0 <= x < w or not 0 <= y < h:
            raise ValueError('Export marker outside original image.')
        state = c['review_status']
        colour = '#ff747b' if state == 'rejected' else '#55e6c2' if state == 'accepted' else '#c595ff' if c.get('border') else '#ff9d42' if c.get('uncertain') else '#ffe05a'
        if review.get('outlines', True):
            for ex, ey in c.get('boundary_px', []):
                if 0 <= ex < w and 0 <= ey < h: draw.point((ex, ey), fill=colour)
        draw.ellipse((x-5, y-5, x+5, y+5), outline=colour, width=1)
        if state == 'rejected':
            draw.line((x-5,y-5,x+5,y+5),fill=colour,width=1)
            draw.line((x-5,y+5,x+5,y-5),fill=colour,width=1)
        draw.text((x+7,y-8),str(c['id']),font=font,fill=colour,stroke_width=1,stroke_fill='black')
        if state == 'accepted' and (not c.get('border') or review.get('include_border')) and roi['x0'] <= x < roi['x1'] and roi['y0'] <= y < roi['y1']: final += 1
    complete = review.get('review_complete', False)
    text = f"{'Final reviewed' if complete else 'Accepted (incomplete)'}: {final}"
    draw.text((5,h+3),text,font=font,fill='white')
    draw.text((5,h+21),f"Border included: {bool(review.get('include_border'))}",font=font,fill='white')
    buf = io.BytesIO(); image.save(buf,format='PNG'); return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def send(self, status, data, kind='application/json'):
        body = data if isinstance(data, bytes) else json.dumps(data, allow_nan=False).encode()
        self.send_response(status); self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store')
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/', '/index.html'): self.send(200, (ROOT/'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/health': self.send(200, {'status': 'ok'})
        else: self.send(404, {'error': 'Not found'})

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 140*1024*1024: raise ValueError('Request empty or too large.')
            r = json.loads(self.rfile.read(length))
            if self.path == '/api/image':
                a, meta = decode_image(r['image'], r.get('green_channel', 1), r.get('blue_channel', 2))
                token = uuid.uuid4().hex
                with LOCK:
                    while len(IMAGES) >= 3: IMAGES.pop(next(iter(IMAGES)))
                    IMAGES[token] = a
                self.send(200, dict(token=token, metadata=meta, png=png(display(a))))
            elif self.path == '/api/count':
                with LOCK: a = IMAGES.get(r['token'])
                if a is None: raise ValueError('Image expired or server restarted. Upload it again.')
                result = count_candidates(a, r.get('options', {}))
                arrays = result.pop('arrays'); previews = {}
                for name, data in arrays.items():
                    if name == 'soma': p = (data > 0).astype(np.uint8)*255
                    elif name == 'processes': p = data.astype(np.uint8)*255
                    else: p = np.clip(data/max(float(np.percentile(data, 99.9)), 1e-12)*255, 0, 255).astype(np.uint8)
                    previews[name] = png(p)
                result['previews'] = previews; self.send(200, result)
            elif self.path == '/api/export-image':
                a, _ = decode_image(r['image'], r.get('green_channel', 1), r.get('blue_channel', 2))
                self.send(200, annotated_png(a, r['review']), 'image/png')
            elif self.path == '/api/validate':
                tolerance = float(r.get('tolerance', 8))
                if not 0 < tolerance <= 100: raise ValueError('Match distance must be 0–100 px.')
                for rows in (r['cells'], r['reference']):
                    if len(rows) > 10000: raise ValueError('Too many reference points.')
                    if any(not np.isfinite(float(c[k])) for c in rows for k in ('x', 'y')): raise ValueError('Invalid coordinates.')
                self.send(200, evaluate_points(r['cells'], r['reference'], tolerance))
            else: self.send(404, {'error': 'Not found'})
        except (ValueError, KeyError, TypeError, OSError, tifffile.TiffFileError) as e:
            self.send(400, {'error': str(e)})
        except Exception as e:
            print(type(e).__name__, str(e), flush=True)
            self.send(500, {'error': 'Processing failed; see server terminal.'})


if __name__ == '__main__':
    host = '127.0.0.1'; port = int(os.environ.get('PORT', '8501'))
    print(f'Histology Workbench: http://{host}:{port}', flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
