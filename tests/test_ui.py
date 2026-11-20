"""Local UI boundaries and full prediction through the existing CLI."""
from io import BytesIO
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

import nibabel as nib
import numpy as np
from PIL import Image
import pytest
import torch

from model.unet3d import UNet3D
from ui.server import Demo, handler_for, inventory, render_slice


def test_axial_overlay_and_bounds():
    ct = np.full((4, 5, 3), -1000., dtype=np.float32)
    mask = np.zeros_like(ct)
    mask[1, 2, 1] = 1
    image = np.asarray(Image.open(BytesIO(render_slice(ct, mask, 1, True))))
    assert image.shape[:2] == (5, 4)
    assert image[2, 1, 0] > image[2, 1, 1] > 0
    assert np.asarray(Image.open(BytesIO(render_slice(ct, mask, 1, False))))[2, 1, 0] == 0
    with pytest.raises(ValueError):
        render_slice(ct, mask, -1, True)


def test_prediction_and_http_boundaries(tmp_path):
    ct = tmp_path / 'ct.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4)), ct)
    checkpoint = tmp_path / 'best.pt'
    torch.save({'model_state': UNet3D(2).state_dict(), 'epoch': 1,
                'config': {'base_channels': 2, 'min_hu': -1000., 'max_hu': 400.,
                           'patch_size': 8, 'threshold': .5}}, checkpoint)
    demo = Demo([(ct, 'val')], [checkpoint], tmp_path / 'output')
    demo.work('open', 0)
    assert demo.state['slices'] == 8
    demo.work('predict', 0)
    assert demo.state['prediction'] and demo.output.is_file()
    assert ct.is_file() and checkpoint.is_file()
    assert np.array_equal(nib.load(demo.output).affine, np.eye(4))
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(demo, 0))
    port = server.server_port
    server.RequestHandlerClass = handler_for(demo, port)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f'http://127.0.0.1:{port}'
    try:
        with urlopen(base + '/api/status') as response:
            assert json.load(response)['prediction']
        with urlopen(base + '/api/download') as response:
            assert response.read() == demo.output.read_bytes()
        for request in [Request(base + '/api/status', headers={'Host': 'evil.example'}),
                        Request(base + '/api/open', data=b'{"choice":0}',
                                headers={'Content-Type': 'application/json', 'Origin': 'http://evil.example'})]:
            with pytest.raises(HTTPError) as error:
                urlopen(request)
            assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        demo.executor.shutdown()


def test_inventory_excludes_test_and_diagnostic(tmp_path):
    rows = ['case_id,patient_id,ct,mask,split']
    for i, split in enumerate(['train', 'val', 'test']):
        for name in ('ct', 'mask'):
            (tmp_path / f'{name}{i}.nii').touch()
        rows.append(f'case{i},patient{i},ct{i}.nii,mask{i}.nii,{split}')
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('\n'.join(rows))
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run/best.pt').touch()
    (tmp_path / 'run/diagnostic.pt').touch()
    scans, checkpoints = inventory(manifest, tmp_path)
    assert [split for _, split in scans] == ['train', 'val']
    assert [p.name for p in checkpoints] == ['best.pt']
