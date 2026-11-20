# Can the model fit a tiny training set?

The one-patient memorization check passed. This is deliberately a **training
fit** check, not generalization or accuracy on unseen patients.

The diagnostic loads only a case marked `train` in the frozen manifest. It
chooses two foreground-centered patches and one negative patch from that case,
keeps them fixed, and repeatedly optimizes the existing architecture. The
negative patch is the densest of up to 64 uniformly sampled foreground-free
training crops, reducing the chance of checking only easy all-air background.
Positive patches may overlap. The target criterion is declared before the run:
each positive patch Dice at least 0.90, and zero false positives on the negative,
with threshold 0.5. Maximum steps are 300; measurements occur every 25 steps.

## November 15 result

Starting from the selected foreground-aware whole-scan checkpoint, this check
passed at step 275 on training case `pilot_01`:

| Fixed training patch | Initial Dice | Final Dice | Final FP | Final FN |
| --- | --- | --- | --- | --- |
| Positive 1 | 0.648549 | 0.910249 | 313 | 11 |
| Positive 2 | 0.581066 | 0.908092 | 357 | 11 |
| Negative | 1.0 | 1.0 | 0 | 0 |

The separate diagnostic checkpoint reloaded and reproduced the same metrics.
No original model weights were overwritten. Validation/test volumes were not
loaded by the command. Complete local results and plots are in
`outputs/experiments/overfit-check1/diagnostic_report.md`.

## Interpretation and next work

The existing network/optimizer can fit this tiny real-data set, so we have
evidence against a gross gradient or target-wiring failure. This does not rule
out every preprocessing/inference issue. It also does not show the model can
locate tumors without annotation-guided crops. The earlier whole-volume
validation Dice remains 0.003449; do not substitute memorization scores for it.

Next inspect full-volume predictions on **training** cases for false-positive
locations; improve coverage with hard negatives and varied crop locations;
expand the patient-level cohort; investigate context and tile boundaries; and
repeat whole-scan validation. Keep the reserved test patient out of selection.

## Reproduce

Choose a new output directory if the following already exists:

```powershell
Set-Location C:\UTampa\CSC401.402\senior-capstone
$py = 'C:\UTampa\CSC401.402\lung-ct-capstone\.venv\Scripts\python.exe'
& $py -m model.overfit_check --manifest data/nlstseg/pilot6/manifest.csv --case-id pilot_01 --initialize outputs/experiments/pilot6-foreground-run1/best.pt --output outputs/experiments/overfit-check1 --steps 300 --threads 4
```

The checkpoint is named `diagnostic.pt` and marked training-only. It is not a
replacement for `best.pt` selected using whole-scan validation. The command also
saves fixed patch tensors, probabilities, sampling seed, source checkpoint hash,
history, and a JSON summary in the ignored output directory.
