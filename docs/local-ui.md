# Local CT viewer

From the repository directory, use the same Python environment as training:

```powershell
cd C:\UTampa\CSC401.402\senior-capstone
$py = 'C:\UTampa\CSC401.402\lung-ct-capstone\.venv\Scripts\python.exe'
& $py -m ui.server
```

Open <http://127.0.0.1:8765>. Stop the server with Ctrl+C in the terminal.
No additional Python packages are required beyond the project requirements.

1. Choose a scan and click **Load scan**. Scroll axial slices using the slider.
2. Choose `pilot6-hardneg-run1` (the default when available).
3. Click **Run 3D U-Net**. CPU inference may take several minutes; the page
   checks progress while the prediction command runs in a background process.
4. Toggle the orange prediction overlay, inspect slices, and download the mask.

The viewer uses the same [prediction command](../model/predict.py), fixed
checkpoint threshold and preprocessing as the command-line workflow. Predictions
and their JSON provenance summaries remain in ignored `outputs/ui/<run>/` folders.
Original scans, expert masks and checkpoints are never overwritten. The interface
does not load expert masks or calculate accuracy; highlighted voxels are not
confirmed tumors. Our best checkpoint still has substantial false positives.

The selector reads training and validation rows from the local pilot manifest.
Test rows are excluded. The page uses generic scan numbers, not patient IDs.
Only `*/best.pt` checkpoints under the checkpoint root are offered; the separate
memorization diagnostic is not offered. To use a different local dataset:

```powershell
& $py -m ui.server --manifest 'data/my-dataset/manifest.csv' --checkpoint-root 'outputs/experiments' --port 8765
```

The manifest must satisfy the existing patient-disjoint manifest validation.
Dataset split provenance remains the user's responsibility. Keep scans and
weights local; they are not distributed with the repository. The server binds
only to loopback, rejects unexpected Host/Origin headers, and uses no external
assets or services. It is a single-user local demo, not a hosted application.
Reloading the page reconnects to the current scan/job. A running prediction
finishes before shutdown completes. Close the server when finished.
