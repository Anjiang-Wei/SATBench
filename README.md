# SATBench: Benchmarking LLMs' Logical Reasoning via Automated Puzzle Generation from SAT Formulas

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](#)
[![arXiv](https://img.shields.io/badge/arXiv-2505.14615-b31b1b.svg)](https://arxiv.org/abs/2505.14615)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/dataset-HuggingFace-orange)](https://huggingface.co/datasets/LLM4Code/SATBench)

---

## Setting Up the Environment

### 1. Clone the repository and enter the directory:

```bash
git clone https://github.com/Anjiang-Wei/SATBench.git
cd SATBench
```

### 2. Create and activate a conda environment:


```bash
conda create -y -n SATBench python=3.12
conda activate SATBench
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Set API keys:

```bash
export OPENAI_API_KEY=<your_openai_api_key>
export ANTHROPIC_API_KEY=<your_anthropic_api_key>
export TOGETHER_API_KEY=<your_together_api_key>
```


## Dataset Generation Pipeline

### 1. Generate SAT/UNSAT CNF Problems

```bash
python step1.1_sat_problem_generation.py \
    --output sat_problems.json \
    --num_per_config 10 \
    --max_clause_len 3 \
    --sat_ratio 0.5 \
    --dimensions "[[3,2],[3,3,2]]" \
    --clauses "[4,5,6,7,8,9,10]"
```

**Key Arguments:**
- `--output`: Output file for SAT problems
- `--num_per_config`: Number of examples per config
- `--dimensions`: Variable dimensionality
- `--clauses`: Clause counts to vary difficulty


### 2. Generate Scenario + Variable Mapping

```bash
python step1.2_scenario_mapping_generation.py \
    --input sat_problems.json \
    --output scenario_and_mapping.jsonl
```

**Key Arguments:**
- `--input`: Input CNF problems
- `--output`: Output JSONL file with generated scenarios and variable meanings


### 3. Generate Final Puzzle Conditions and Questions

```bash
python step1.3_puzzle_generation.py \
    --input scenario_and_mapping.jsonl \
    --output puzzle_problems.jsonl \
    --target_per_clause 5 \
    --clause_values "[4,5,6]"
```

**Key Arguments:**
- `--input`: Scenario and mapping file
- `--output`: Final puzzle output
- `--target_per_clause`: Number of examples to generate per clause count
- `--clause_values`: Which clause numbers to include in the final benchmark


You can also access the released version on Hugging Face:

👉 [https://huggingface.co/datasets/LLM4Code/SATBench](https://huggingface.co/datasets/LLM4Code/SATBench)

Each puzzle includes:
- A natural language **scenario + variable mapping**
- A list of **logical conditions**
- A **final question**
- Ground-truth **SAT/UNSAT** label

## Evaluation Pipeline

We provide scripts to evaluate LLMs on SATBench directly from the Hugging Face dataset.

### Running Evaluations

```bash
python step2_evaluation.py \
    --mode sat \
    --eval_model openai_gpt-4o \
    --limit 10
```

```bash
python step2_evaluation.py \
    --mode trace \
    --eval_model openai_gpt-4o
```

**Modes**:
- `--mode sat`: Run SAT/UNSAT prediction for each puzzle.
- `--mode trace`: Judge reasoning trace quality (after SAT/UNSAT evaluation).

**Arguments:**
- `--limit`: Max number of examples to evaluate (optional).
- `--eval_model`: Model used to answer puzzles (e.g., `openai_gpt-4o`, `together_Qwen3-14B`).


### Statistics Summary

After evaluation, generate per-bucket statistics:

```bash
python step3_stats.py
```

Or for specific models:

```bash
python step3_stats.py --models openai_gpt-4o
```

This prints accuracy by SAT/UNSAT and difficulty (easy/medium/hard)


## Supported Models

SATBench supports evaluation for a wide range of models via the OpenAI, Anthropic, and Together APIs:

- `openai_gpt-4o`
- `openai_gpt-4o-mini`
- `openai_o4-mini`
- `together_deepseek-ai_DeepSeek-R1`
- `together_deepseek-ai_DeepSeek-V3`
- `anthropic_claude-3-7-sonnet-20250219`
- `together_Qwen_Qwen3-235B-A22B-fp8-tput`
- `together_Qwen_QwQ-32B`
- `together_deepseek-ai_DeepSeek-R1-Distill-Qwen-14B`
- `together_meta-llama_Llama-4-Scout-17B-16E-Instruct`
- `together_meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8`
- `together_meta-llama_Llama-3.3-70B-Instruct-Turbo`
- `together_meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo`
- `together_meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo`


## Citation

If you use this benchmark in your research, please cite:

```
@article{wei2025satbench,
  title={SATBench: Benchmarking LLMs' Logical Reasoning via Automated Puzzle Generation from SAT Formulas},
  author={Wei, Anjiang and Wu, Yuheng and Wan, Yingjia and Suresh, Tarun and Tan, Huanmi and Zhou, Zhanke and Koyejo, Sanmi and Wang, Ke and Aiken, Alex},
  journal={arXiv preprint arXiv:2505.14615},
  year={2025}
}
```



