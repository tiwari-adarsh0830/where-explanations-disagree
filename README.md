# Where Explanations Disagree

Final project for DS-GA 3001 Visualization for Machine Learning (NYU, Fall 2026).
Team 7: Sheetal Sood and Adarsh Tiwari.

## What we're doing

Explanation methods for language models don't always agree. Attention maps, logit attribution and gradient methods can point at different attention heads for the same prediction. We want to build a visual analytics tool that shows where they disagree, and use activation patching (knocking out a head or a group of heads and checking what happens to the output) as a check on which explanation holds up.

We're starting with GPT-2 small on the indirect object identification (IOI) task. Wang et al. (2023) already mapped the circuit GPT-2 uses for it, so we have something to compare against.

**Status:** in progress. The proposal was approved in October. Pipeline v3 and the first round of analysis are done, and the D3 tool is next.

## What's in here

| Folder | What it has |
| --- | --- |
| `pipeline/` | Python scripts that compute the explanations and patching effects |
| `data/` | `ioi_lists.json`: templates, names, places and objects from Wang et al.'s dataset code |
| `results/` | Output of our first proper run (raw JSON and a per-head table) |
| `frontend/` | The D3 tool. Not started yet |
| `docs/` | Links to our planning docs |

In `pipeline/`, `ioi_check_v3.py` is the current pipeline. It's the same as v2 (120 prompts from the paper's templates) but it keeps every prompt's scores, labels each prompt with its template and name order, and records attention at five query/key positions instead of one. `analysis_v3.py` reads that output and does the bootstrap intervals, the per-order and per-template breakdowns, and the attribution patching error analysis. `ioi_check_v2.py` produced the numbers in our proposal and `ioi_smoke_test_v1.py` was our first rough try. We kept both for reference.

## Setup

We ran this on Python 3.13 with CPU PyTorch.

```
pip install -r requirements.txt
```

TransformerLens 4.x removed `HookedTransformer`, which the IOI code and most tutorials use, so `requirements.txt` pins it below 4.

## Running it

```
python pipeline/ioi_check_v3.py data/ioi_lists.json results/ioi_v3_results.json
python pipeline/analysis_v3.py results/ioi_v3_results_per_prompt.json results/analysis_v3.json
```

GPT-2 small downloads from Hugging Face the first time. The pipeline takes about 4.5 minutes on a MacBook CPU (about 19 on a 2 core cloud machine) because patching every head one at a time is slow. The analysis takes a few seconds. v3's averages match v2 exactly, so the proposal numbers still hold.

## First results (one run, so preliminary)

- GPT-2 small got all 120 prompts right (mean logit difference 4.19).
- Across the 144 heads, attention agreed only weakly with single-head patching (Spearman 0.17). Logit attribution got 0.36 and attribution patching 0.88.
- Attention ranks the S-inhibition heads 23rd to 69th. Patching puts them 2nd to 9th.
- Knocking out the 3 name movers drops the logit difference by only 0.11 on average. That average hides a split though: it drops by 0.67 when the subject comes first (BABA) and goes up by 0.45 when the indirect object comes first (ABBA). Knocking them out together with the 8 backup name movers drops it by 2.66.

So attention misses part of the circuit, and patching one head at a time misses the backup heads. That's why we want group patching in the tool.

Since then we found that attention mostly misses those heads because we only looked at the final token. Scoring each head where it actually looks (for example the S-inhibition heads looking at the second name) raises its agreement with patching from 0.17 to 0.42. Bootstrap intervals and the rest are in `results/analysis_v3.json`.

## Main references

- Wang et al. 2023, *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*
- Vig 2019, *A Multiscale Visualization of Attention in the Transformer Model* (BertViz)
- Silva et al. 2025, *Visagreement: Visualizing and Exploring Explanations (Dis)Agreement* (IEEE VIS)
- Vaidyanathan et al. 2026, *The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching*

## AI use

Following the course AI policy: both scripts in `pipeline/` were written with help from Claude (an AI assistant). We're going through them and re-running them ourselves. We'll keep this section updated as the project goes.
