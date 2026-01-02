# Codex Agent Evaluation on SATBench

This directory contains scripts to evaluate the Codex agent on the SATBench logical reasoning benchmark.

## Prerequisites

1. **Codex CLI installed**: Run the setup script to install Codex
   ```bash
   ./codex-setup.sh
   ```

2. **Python dependencies**: Install required packages
   ```bash
   pip install -r requirements.txt
   ```

3. **API Key**: Set your API key as an environment variable
   ```bash
   export OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### Basic Usage (100 tasks, default model)

```bash
python evaluate_codex.py
```

### Full Dataset Evaluation with specified model

```bash
python evaluate_codex.py --model gpt-4o --limit 2100
```

## Command-Line Arguments

- `--model`: Model to use with codex (default: `gpt-4o`)
- `--limit`: Maximum number of samples to evaluate (default: `10`)
- `--output`: Path to save evaluation results (default: `codex_eval_results.jsonl`)
- `--n-concurrent`: Number of concurrent evaluations (default: `20`)

## Example Output

```
Loading SATBench dataset...
Evaluating 10 tasks with codex (model: gpt-4o)
100%|████████████████████████████████████| 10/10 [01:23<00:00,  8.35s/it]

FINAL ACCURACY: 70.00% (7/10)
```

## Citation

If you use this evaluation script, please cite the original SATBench paper:

```bibtex
@article{wei2025satbench,
  title={SATBench: Benchmarking LLMs' Logical Reasoning via Automated Puzzle Generation from SAT Formulas},
  author={Wei, Anjiang and Wu, Yuheng and Wan, Yingjia and Suresh, Tarun and Tan, Huanmi and Zhou, Zhanke and Koyejo, Sanmi and Wang, Ke and Aiken, Alex},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={33820--33837},
  year={2025}
}
```
