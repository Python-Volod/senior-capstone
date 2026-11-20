"""Serve the local research demo: python -m ui.server."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import threading
from urllib.parse import parse_qs, urlparse
import uuid

import numpy as np
from matplotlib.image import imsave

from model.experiment import read_manifest
from preprocessing.load_ct import load_volume, validate_pair

ROOT = Path(__file__).resolve().parents[1]


def inventory(manifest, checkpoint_root):
    # Do not expose patient identifiers or offer the reserved test scans.
    scans = [(Path(row['ct']), row['split']) for row in read_manifest(manifest)
             if row['split'] in {'train', 'val'}]
    checkpoints = sorted(Path(checkpoint_root).glob('*/best.pt'))
    return scans, checkpoints


def render_slice(ct, mask, index, overlay):
    """Show canonical axial XY as rows Y/columns X, anterior at the top."""
    if index < 0 or index >= ct.shape[2]:
        raise ValueError('Slice is outside this scan.')
    gray = np.clip((ct[:, :, index].T[::-1] + 1000.) / 1400., 0, 1)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if overlay and mask is not None:
        selected = mask[:, :, index].T[::-1] > 0
        rgb[selected] = .6 * rgb[selected] + .4 * np.array([1., .28, .12])
    output = BytesIO()
    imsave(output, rgb, format='png')
    return output.getvalue()


class Demo:
    def __init__(self, scans, checkpoints, output_root):
        self.scans, self.checkpoints = scans, checkpoints
        self.output_root = Path(output_root)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.ct = self.mask = self.scan = self.output = None
        self.state = {'status': 'empty', 'message': 'Choose a scan to begin.'}

    def start(self, action, choice):
        with self.lock:
            if self.state['status'] == 'busy':
                raise ValueError('An operation is already running.')
            options = self.scans if action == 'open' else self.checkpoints
            if type(choice) is not int or not 0 <= choice < len(options):
                raise ValueError('Choose an available scan or checkpoint.')
            if action == 'predict' and self.ct is None:
                raise ValueError('Load a scan first.')
            self.state = {'status': 'busy', 'message': 'Loading scan…' if action == 'open'
                          else 'Running 3D U-Net on CPU. This can take several minutes…'}
            self.executor.submit(self.work, action, choice)

    def work(self, action, choice):
        try:
            if action == 'open':
                ct = load_volume(self.scans[choice][0])
                with self.lock:
                    self.ct, self.scan = ct, choice
                    self.mask = self.output = None
                    self.state = {'status': 'ready', 'message': 'Scan loaded. Ready for prediction.',
                                  'slices': ct.shape[2], 'shape': list(ct.shape), 'scan_index': choice}
            else:
                # Reuse the CLI in a separate process: no alternate model/preprocessing.
                output = self.output_root / uuid.uuid4().hex / 'prediction.nii.gz'
                result = subprocess.run([sys.executable, '-m', 'model.predict', '--ct',
                                         str(self.scans[self.scan][0]), '--checkpoint',
                                         str(self.checkpoints[choice]), '--output', str(output)],
                                        cwd=ROOT, capture_output=True, text=True)
                if result.returncode:
                    raise ValueError('Prediction failed. Check the local terminal for details.')
                mask, metadata = load_volume(output, return_metadata=True)
                _, source_metadata = load_volume(self.scans[self.scan][0], return_metadata=True)
                validate_pair(self.ct, mask, source_metadata, metadata)
                summary = json.loads(Path(str(output) + '.json').read_text())
                with self.lock:
                    self.mask, self.output = mask, output
                    self.state = {'status': 'ready', 'message': 'Prediction complete.',
                                  'slices': self.ct.shape[2], 'shape': list(self.ct.shape),
                                  'scan_index': self.scan, 'checkpoint': self.checkpoints[choice].parent.name,
                                  'prediction': True, 'voxels': summary['foreground_voxels'],
                                  'seconds': round(summary['network_inference_seconds'], 1),
                                  'epoch': summary['checkpoint_epoch'], 'threshold': summary['threshold']}
        except Exception as error:
            print(f'Demo operation failed: {error}', file=sys.stderr)
            if action == 'predict' and 'result' in locals() and result.returncode:
                print(result.stderr, file=sys.stderr)
            with self.lock:
                if action == 'open':
                    self.ct = self.mask = self.scan = self.output = None
                self.state = {'status': 'error', 'message': 'Could not load this scan.' if action == 'open'
                              else 'Prediction failed; see the local terminal. Reload a scan to retry.'}


def handler_for(demo, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, body, content_type='application/json', status=200, download=False):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; frame-ancestors 'none'")
            if download:
                self.send_header('Content-Disposition', 'attachment; filename="prediction.nii.gz"')
            self.end_headers()
            self.wfile.write(body)

        def trusted(self):
            allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}
            origin = self.headers.get('Origin')
            return (self.headers.get('Host') in allowed and
                    (origin is None or origin in {f'http://{host}' for host in allowed}))

        def do_GET(self):
            if not self.trusted():
                return self.send(b'{}', status=403)
            url = urlparse(self.path)
            try:
                if url.path in {'/', '/app.js', '/style.css'}:
                    name = 'index.html' if url.path == '/' else url.path[1:]
                    kind = {'index.html': 'text/html; charset=utf-8', 'app.js': 'text/javascript',
                            'style.css': 'text/css'}[name]
                    return self.send((Path(__file__).parent / name).read_bytes(), kind)
                with demo.lock:
                    if url.path == '/api/catalog':
                        value = {'scans': [f'Scan {i+1:02} · {split}' for i, (_, split) in enumerate(demo.scans)],
                                 'checkpoints': [p.parent.name for p in demo.checkpoints]}
                        return self.send(json.dumps(value).encode())
                    if url.path == '/api/status':
                        return self.send(json.dumps(demo.state).encode())
                    if url.path == '/api/slice' and demo.ct is not None and demo.state['status'] != 'busy':
                        query = parse_qs(url.query)
                        return self.send(render_slice(demo.ct, demo.mask, int(query.get('z', ['0'])[0]),
                                                      query.get('overlay', ['0'])[0] == '1'), 'image/png')
                    if url.path == '/api/download' and demo.output and demo.state['status'] != 'busy':
                        return self.send(demo.output.read_bytes(), 'application/gzip', download=True)
                self.send(b'{}', status=404)
            except (ValueError, OSError):
                self.send(b'{"message":"Request could not be completed."}', status=400)

        def do_POST(self):
            if not self.trusted() or self.headers.get('Content-Type') != 'application/json':
                return self.send(b'{}', status=403)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length < 1024:
                    raise ValueError('Invalid request size.')
                if self.path not in {'/api/open', '/api/predict'}:
                    return self.send(b'{}', status=404)
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError('Invalid request.')
                demo.start(self.path.rsplit('/', 1)[1], body.get('choice'))
                self.send(b'{}', status=202)
            except (ValueError, TypeError) as error:
                self.send(json.dumps({'message': str(error)}).encode(), status=400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/nlstseg/pilot6/manifest.csv')
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT / 'outputs/experiments')
    args = parser.parse_args()
    scans, checkpoints = inventory(args.manifest, args.checkpoint_root)
    demo = Demo(scans, checkpoints, ROOT / 'outputs/ui')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(demo, args.port))
    print(f'Local research demo: http://127.0.0.1:{args.port} (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        demo.executor.shutdown(wait=True)


if __name__ == '__main__':
    main()
