Regenerated big-model rectification results

Files:
- qwen235b_v4/: regenerated v4 rectification outputs for qwen-3-235b-a22b-instruct-2507
- llama31_8b_v5/: regenerated v5 rectification outputs for llama3.1-8b
- validation_report.json: byte-level comparison against archived v10 summary files

Validation outcome:
* qwen235b_v4: byte_identical=True, generated_sha256=e4fa55dcf6db3bf42f0483d66bb074c4173677aa354ad3cf393721ffa2b8086c
  test baseline=0.179387, corrected=0.176710, delta=-0.002677
* llama31_8b_v5: byte_identical=False, generated_sha256=5f7b0be0169340f62a52dc90c8b01632e080ff0e4fb553cfebfa81e02793c406
  test baseline=0.206104, corrected=0.191543, delta=-0.014561
