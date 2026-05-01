# Missing Information to Confirm Before Submission

This file lists information that could not be fully verified from the local codebase, university instruction files, existing drafts and research folders. These items were intentionally kept out of the main report body as unresolved markers.

## Administrative Details

- Confirm the official final submission date to use on the title page. The generated report currently uses `April 2026` as the draft date.
- Confirm whether the School requires a specific official wording for the GenAI Tools Accountability Statement. The report includes a clear accountability statement, but no exact mandated wording was found in the extracted instruction text.

## Dataset and Ethics Details

- Confirm the exact original source of the local evaluation dataset in `datasets/gallery`, `datasets/test` and `datasets/test_masked`.
- Confirm the dataset licence and whether the dataset is permitted in final supporting material.
- Confirm whether any public-figure names/images in demo data or evaluation material should be replaced, anonymised or excluded from the final submission package.
- Confirm the exact cleaning procedure used to create the 366-identity gallery/test split.
- Confirm the exact method used to generate `datasets/test_masked`, including whether masks were synthetic, which tool/template was used, and whether all identities/images were transformed consistently.

## Screenshots and Report Evidence

- Insert final screenshots manually before submission:
  - backend `/health`;
  - kiosk scanning state;
  - recognised traveller dashboard;
  - passport/boarding/flight display;
  - prototype map/directions;
  - selected evaluation plots.
- Confirm whether screenshots should be placed in Chapter 4/5, Appendix E, or both.

## Citation Details to Check

- `Bhatia et al. (n.d.)`: the local PDF provides the title and authors but no verified year or venue in extracted text.
- `Erhart (1993)`: the existing draft/source summary provides the year, but the extracted PDF text did not expose a full publication venue.
- `AdaFace`, `FocusFace` and `MaskInv`: the citation details were verified from local external repository README files rather than standalone PDFs in `FYP resrarchs`. Add the original papers to the research folder if your supervisor expects every model citation to come from the research folder.
- Check final Harvard formatting for arXiv sources and whether access dates are required by the university style guide.

## Verification and Formatting

- LibreOffice/`soffice` was not available on this machine, so rendered page-image QA of `FINAL_REPORT_USING_TEMPLATE.docx` could not be completed with the document rendering tool.
- Open `FINAL_REPORT_USING_TEMPLATE.docx` in Microsoft Word, update the table of contents field, and visually check page breaks, table wrapping and figure placeholders before submission.
- Run Word spellcheck/grammar check after inserting screenshots.

## Supporting Material Decisions

- Confirm whether `evaluation/results_*.csv` and `evaluation/plots/` should be included as supporting evidence.
- Confirm whether model weights and generated FAISS databases should be submitted, shared separately, or excluded because of file size/licensing.
- Confirm whether `external/` should be included in the submission package or only referenced as external/vendor code.
- Keep `SmartIdentity/trash` as archive/evidence only unless your supervisor explicitly asks for specific historical files.

