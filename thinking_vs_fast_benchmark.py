import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import pandas as pd

# Load a free model as placeholder (GPT-2); replace with Gemma 4 when available
model_name = "gpt2"
print(f"Loading model {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# Function to generate response
def generate_response(prompt, thinking=False, max_length=512):
    if thinking:
        prompt = f"Think step by step: {prompt}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Define test cases
test_cases = [
    {
        "type": "logic_puzzle",
        "prompt": "There are 12 balls, one of which is heavier or lighter. Using a balance scale, find the odd ball in 3 weighings.",
        "description": "Classic balance puzzle"
    },
    {
        "type": "code_debug",
        "prompt": "Debug this Python function: def factorial(n): if n == 0: return 1 else: return n * factorial(n-1). It works for positive n but fails for negative. Fix it.",
        "description": "Factorial function debug"
    },
    {
        "type": "logic_puzzle",
        "prompt": "A lily pad doubles in size every day. It covers the pond in 30 days. When was the pond half covered?",
        "description": "Lily pad growth puzzle"
    }
]

# Thinking mode
thinking_results = []
for case in test_cases:
    prompt = case["prompt"]
    response = generate_response(prompt, thinking=True)
    if "Answer:" in response:
        thinking = response.split("Answer:")[0].replace("Think step by step: ", "").strip()
        final_answer = response.split("Answer:")[1].strip()
    else:
        thinking = "No thinking captured"
        final_answer = response
    thinking_results.append({
        "case": case["description"],
        "thinking": thinking,
        "final_answer": final_answer
    })
    print(f"Thinking for {case['description']}: {thinking[:180]}...")

# Standard mode
standard_results = []
for case in test_cases:
    prompt = case["prompt"]
    response = generate_response(prompt, thinking=False)
    standard_results.append({"case": case["description"], "answer": response})
    print(f"Standard for {case['description']}: {response[:180]}...")

# Compare outputs
comparisons = []
for think, std in zip(thinking_results, standard_results):
    diff = think["final_answer"] != std["answer"]
    corrections = len(re.findall(r'correct|wrong|mistake', think["thinking"].lower()))
    comparisons.append({
        "case": think["case"],
        "thinking_answer": think["final_answer"],
        "standard_answer": std["answer"],
        "differ": diff,
        "corrections_in_thinking": corrections
    })

# Display comparisons
df = pd.DataFrame(comparisons)
print("\nComparison table:")
print(df)

# Create plotly chart
fig = make_subplots(rows=1, cols=2, subplot_titles=("Differences in Answers", "Corrections in Thinking"))
fig.add_trace(go.Bar(x=df["case"], y=df["differ"].astype(int), name="Answers Differ"), row=1, col=1)
fig.add_trace(go.Bar(x=df["case"], y=df["corrections_in_thinking"], name="Corrections Count"), row=1, col=2)
fig.update_layout(title_text="Thinking vs Standard Mode Comparison")
fig.write_html("thinking_vs_fast_benchmark.html")
print("Saved plot to thinking_vs_fast_benchmark.html")
