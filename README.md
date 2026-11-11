
### Repository Structure
```
PRM-guided-Best-of-N-search/
|-- data/
|   `-- gsm8k_test_50.jsonl          # Fixed 50-problem evaluation benchmark
|-- src/
|   |-- __init__.py
|   |-- generation.py                # Step prompt, greedy, self-consistency
|   |-- extraction.py                # Regex parser for steps and numerical answers
|   |-- prm_judge.py                 # Step-level verifier prompt & scoring loop
|   |-- search.py                    # PRM-guided Best-of-N and budget tracking
|   `-- utils.py                     # Dual-model loading, generate/judge callables, bookkeeping helpers
|-- tests/
|   |-- __init__.py
|   |-- test_public_adapter_smoke.py # Public smoke test suite for grading interface
|   `-- test_token_counting.py       # Offline token count test with batch padding
|-- outputs/                         # Generated predictions and metric dumps
|   |-- results_summary.json         # Automated metrics table across all 6 configurations
|   |-- aggregation_ablation.json    # Aggregation strategies and lambda sweep ablation
|   |-- sc_scaling_curve.json        # Self-consistency scaling curve (N=1..15)
|   `-- per_problem.jsonl            # Detailed per-problem records and step scores
|-- run_pipeline.ipynb               # End-to-end evaluation pipeline notebook (you create this)
|-- submission_adapter.py            # Grading adapter contract (Required)
|-- requirements.txt
|-- pyproject.toml
|-- make_submission.sh               # Submission bundler script
|-- REPORT_TEMPLATE.md               # Scaffold for written report
`-- README.md
```

