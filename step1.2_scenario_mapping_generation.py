"""
SAT Scenario and Variable Mapping Generator
=========================================================================

This script takes a JSON file of SAT problems and uses
GPT to generate corresponding natural language scenarios and variable explanations.

Usage Example:
--------------
python step1.2_scenario_mapping.py \
    --input sat_problems.json \
    --output scenario_and_mapping.jsonl

Arguments:
----------
--input:  Path to input JSON file containing SAT problems.
--output: Path to write generated JSONL entries (scenario + mapping).
"""
import json
import asyncio
import os
import re
import yaml
import argparse
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# === CONFIG ===
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="sat_problems.json")
parser.add_argument("--output", type=str, default="scenario_and_mapping.jsonl")
args = parser.parse_args()

INPUT_FILE = args.input
OUTPUT_FILE = args.output
MODEL = "gpt-4o"
semaphore = asyncio.Semaphore(80)

# === OpenAI client using environment variable ===
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# === Load prompt template from YAML ===
def load_prompt(path="prompts/gen_prompts.yaml", key="scenario_mapping"):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data[key]["system"]

PROMPT_TEMPLATE = load_prompt()

# === Async GPT call ===
async def ask_gpt_generate_scenario(readable_formula):
    prompt = PROMPT_TEMPLATE.strip().replace("{readable}", readable_formula)
    try:
        res = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"[GPT failed] {e}")
        return None

# === I/O Writer ===
def write_jsonl(path, item):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()

# === Load existing scenarios ===
def load_existing_keys():
    if not os.path.exists(OUTPUT_FILE):
        return set()
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        return set(json.loads(line).get("readable") for line in f)

# === Per-entry async worker ===
async def generate_one(entry):
    async with semaphore:
        readable = entry.get("readable")
        if not readable:
            return

        result = await ask_gpt_generate_scenario(readable)
        if not result:
            return

        scenario_match = re.search(r"<scenario description>\s*(.*?)\s*<variable explanation>", result, re.DOTALL | re.IGNORECASE)
        mapping_match = re.search(r"<variable explanation>\s*(.*?)$", result, re.DOTALL | re.IGNORECASE)

        if not (scenario_match and mapping_match):
            print(f"[Warning] Failed to parse tags from:\n{result}\n")
            return

        entry["scenario"] = scenario_match.group(1).strip()
        entry["variable_mapping"] = mapping_match.group(1).strip()
        await asyncio.to_thread(write_jsonl, OUTPUT_FILE, entry)

# === Main parallel driver ===
async def generate_scenarios():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        problems = json.load(f)

    existing_keys = load_existing_keys()
    problems = [p for p in problems if p.get("readable") not in existing_keys]

    print(f"Remaining to generate: {len(problems)}")

    tasks = [generate_one(entry) for entry in problems]
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await fut

# === Launch ===
if __name__ == "__main__":
    asyncio.run(generate_scenarios())