# Tools

This folder contains local tooling used to prepare and support the SmartIdentity evaluation evidence. These scripts are separate from the live kiosk runtime.

## `dataset_preparation/`

APFF dataset preparation, identity-aware cleaning, alignment and gallery/test split scripts. These files support Appendix B and Chapter 5 evidence, but the final reported evaluation numbers come from the saved outputs in `evaluation/`.

- `prepare_apff.py` prepares APFF-style source folders and aligned images.
- `align_by_identity.py` performs identity-aware cleaning and alignment.
- `split_dataset.py` creates gallery/test folders from aligned images.

## `mask_generation/`

Synthetic masked-query generation tooling used to prepare the masked test set.

- `add_mask.py` applies the adapted MaskInv-style mask mapping to ArcFace-aligned 112 x 112 test images.
- `assets/mask_img.png` is the local mask template used by `add_mask.py`.

## `archive/`

Older or experimental scripts retained for provenance.

- `eval_apff.py` is an older APFF evaluation helper and is not used for the final Chapter 5 reported results.

Do not regenerate or change final evaluation numbers from archived scripts. The final report evaluation numbers come from the saved CSVs and plots under `evaluation/`.

The project-level `utils/` folder remains outside `tools/` and is the single shared helper location for runtime backend, model wrappers, evaluation code and tool scripts.
