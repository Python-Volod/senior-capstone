# Integration sprint: response to November professor feedback

The professors recognized technical progress, reflection, ethics, and the UI
direction. The next deliverable should add measured model evaluation, related
research, an explicit project contribution, and visible task ownership.

## Proposed contribution

Investigate practical trade-offs between small 2D and 3D U-Nets for lung lesion
segmentation on a shared patient-disjoint dataset, with a simple interface to
review predictions. Measure segmentation quality and runtime; investigate memory
if resources permit. Better accuracy, speed, or usability is a hypothesis to
test, not an established advantage. The current 3D model is a narrow educational
baseline; no claim of architectural novelty or clinical utility is made.

## Workstreams and acceptance criteria

Assign actual team members at the next meeting; these roles are proposals.

| Workstream | Role to assign | Completion evidence |
| --- | --- | --- |
| Data and splits | Data owner | Verified patient identities, label definitions, frozen manifest, adequate cohort |
| 3D baseline | Model owner | Saved checkpoint, reproducible training settings, held-out whole-scan report |
| 2D baseline | Teammate/model owner | Shared code and checkpoint; complete test masks on the common grid |
| Evaluation/research | Evaluation owner | Comparison CSV, failure examples, related-work table with task differences |
| UI integration | Interface owner | Load case, show clearly distinguished prediction/expert overlay, recorded processing scope |

Use weekly tasks with one named owner, acceptance criteria, and visible updates.
Address the absent-member/contact issue with the team and course process rather
than silently assigning their work to someone else. Do not infer other members'
effort from a missing commit or the reported peer-feedback percentage alone.

## Related research supplied by professors

| Source | Task | Role in this project |
| --- | --- | --- |
| [Google/Ardila et al., 2019](https://www.nature.com/articles/s41591-019-0447-x) | Lung cancer screening with 3D deep learning | Related work; different endpoint from voxel lesion masks |
| [MONAI lung nodule CT detection](https://huggingface.co/MONAI/lung_nodule_ct_detection/blob/main/docs/README.md) | LUNA16-trained 3D RetinaNet detection | Study pretrained detection workflow; not a directly interchangeable segmentation baseline |
| [Duke benchmarking repository](https://github.com/fitushar/AI-in-Lung-Health-Benchmarking-Detection-and-Diagnostic-Models-Across-Multiple-CT-Scan-Datasets) | Nodule detection and malignancy classification | Study reproducible evaluation, dataset effects, sensitivity versus FP/scan |

These sources were inspected in the feedback discussion. Before reproducing a
specific model, inspect its pinned version, weights, dependencies, training
cohort, license, and output definitions. A shared dataset name is not enough:
test patients must not overlap its pretraining/training data. Do not compare a
screening AUC, detection sensitivity at FP/scan, and segmentation Dice as though
they measure the same outcome. Detection needs a separate matching protocol if
the team later expands scope beyond masks.

## Sequence and current blockers

1. Acquire and verify more CT/expert-mask pairs and real patient grouping.
2. Agree on the common target, splits, metrics, and resource budget.
3. Train the 3D baseline; obtain the teammate's reproducible 2D implementation.
4. Freeze both using validation, then evaluate held-out cases once for reporting.
5. Review false positives/negatives, limitations, and measured compute trade-offs.
6. Connect the selected checkpoint to a minimal review UI.

At initial workflow implementation, only one extracted real pair was known
locally, and no 2D model was found in the fetched team branches. As of November
15, a six-patient tumor-only pilot is prepared for initial training; a larger
cohort and teammate code are still needed for a meaningful model comparison.
UI work should clearly distinguish expert masks from model
predictions and should not display invented confidence or accuracy numbers.
