# Gemma 4 Thinking vs Fast Reasoning Benchmark

This project benchmarks Gemma 4's "Thinking" mode against its "Fast" (standard) reasoning on complex logic puzzles and code debugging tasks. It visualizes the differences in reasoning depth and self-correction.

## Features
- Side-by-side comparison of model outputs
- Visualization of thinking traces and corrections
- Support for logic puzzles and code debugging tasks
- Interactive dashboard (via Jupyter notebook)

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the notebook: `jupyter notebook thinking_vs_fast_benchmark.ipynb`

## Usage
- Load the Gemma 4 2B model
- Define tasks (puzzles or code bugs)
- Generate responses in both modes
- Visualize results

## Model
Using GPT-2 as a free demo model (simulating thinking with "Think step by step" prompts). Replace with `"google/gemma-4-2b-it"` when Gemma 4 is released and accessible.