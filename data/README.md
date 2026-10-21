# Local dataset files

Datasets are downloaded separately under their access and license terms. No
dataset is included. Keep all scans, annotations, manifests, and derived images
local; only this README is tracked under data/.

For the first real test, provide a de-identified scalar 3D CT NIfTI file and its
matching expert lesion segmentation NIfTI file, for example:

```
data/sample/ct.nii.gz
data/sample/mask.nii.gz
```

These are example local names, not a claimed NLSTseg directory layout. The mask
must describe this exact CT and share its physical grid. Obtain the provider's
label definitions (which IDs represent lesions), intensity units/scaling, and
voxel spacing units. The loader assumes spatial units are millimeters; verify
this from the source. Positive labels are highlighted for initial inspection;
they are not automatically converted into binary lesion ground truth.

If the download instead contains DICOM, RTSTRUCT, DICOM SEG, or separate reader
annotations, provide a de-identified folder tree, file extensions, and the
provider's format/label documentation first. Do not rename these files to .nii.
Dataset-specific pairing, conversion, or reader consensus requires inspection.
Do not include patient identifiers in filenames, screenshots, or shared logs.
