# First real-data candidate: NLSTseg

The [authors' paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12375103/)
describes expert lesion annotations paired with low-dose CT scans in NIfTI
format. Its documented filename pattern is `<ID>_CT.nii.gz` with the matching
`<ID>_tumor.nii.gz`. Keep actual identifiers out of shared examples and logs.

The [Zenodo release](https://zenodo.org/records/14838349) contains a small table
archive and six image archives totaling approximately 34.2 GB. One image
archive, not the entire collection, is sufficient to obtain an initial pair.

The paper distinguishes annotation IDs from tumor/nodule type in its label
table. Preserve voxel labels and inspect that table before choosing targets.

No real scan has been downloaded or tested in this branch. After obtaining
one pair, load both with metadata, validate their grids, inspect an overlay,
and check tensor shapes. Verify source units/scaling and dataset usage terms.
Store all downloaded/extracted material under ignored `data/`; commit none
of it. These notes do not describe the separate IDC DICOM SEG distribution.
