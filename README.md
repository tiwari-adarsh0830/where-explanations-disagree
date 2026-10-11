# Where Explanations Disagree

Final project for DS-GA 3001 Visualization for Machine Learning (NYU, Fall 2026).
Team 7: Sheetal Sood and Adarsh Tiwari.

## What we're doing

Explanation methods for language models don't always agree. Attention maps, logit attribution and gradient methods can point at different attention heads for the same prediction. We want to build a visual analytics tool that shows where they disagree, and use activation patching (knocking out a head or a group of heads and checking what happens to the output) as a check on which explanation holds up.

We're starting with GPT-2 small on the indirect object identification (IOI) task. Wang et al. (2023) already mapped the circuit GPT-2 uses for it, so we have something to compare against.

**Status:** early. The proposal is due Oct 20 and we're still waiting on mentor approval for the custom project.

## What's in here

| Folder | What it has |
| --- | --- |
| `pipeline/` | Python scripts that compute the explanations and patching effects |
| `data/` | `ioi_lists.json`: templates, names, places and objects from Wang et al.'s dataset code |
| `results/` | Output of our first proper run (raw JSON and a per-head table) |
| `frontend/` | The D3 tool. Not started yet |
| `docs/` | Links to our planning docs |

In `pipeline/`, `ioi_check_v2.py` is the current check (120 prompts from the paper's templates). `ioi_smoke_test_v1.py` was our first rough try with our own templates. We kept it for reference but the v2 numbers are the ones to use.

## Setup

We ran this on Python 3.13 with CPU PyTorch.

```
pip install -r requirements.txt
```

TransformerLens 4.x removed `HookedTransformer`, which the IOI code and most tutorials use, so `requirements.txt` pins it below 4.

## Running it

```
python pipeline/ioi_check_v2.py data/ioi_lists.json results/ioi_v2_results.json
```

GPT-2 small downloads from Hugging Face the first time. The full run took about 19 minutes on 2 CPU cores because patching every head one at a time is slow. Colab with a GPU should be a lot faster.

## First results (one run, so preliminary)

- GPT-2 small got all 120 prompts right (mean logit difference 4.19).
- Across the 144 heads, attention agreed only weakly with single-head patching (Spearman 0.17). Logit attribution got 0.36 and attribution patching 0.88.
- Attention ranks the S-inhibition heads 23rd to 69th. Patching puts them 2nd to 9th.
- Knocking out the 3 name movers drops the logit difference by only 0.11. Knocking them out together with the 8 backup name movers drops it by 2.66.

So attention misses part of the circuit, and patching one head at a time misses the backup heads. That's why we want group patching in the tool.

## Main references

- Wang et al. 2023, *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*
- Vig 2019, *A Multiscale Visualization of Attention in the Transformer Model* (BertViz)
- Silva et al. 2025, *Visagreement: Visualizing and Exploring Explanations (Dis)Agreement* (IEEE VIS)
- Vaidyanathan et al. 2026, *The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching*

## AI use

Following the course AI policy: both scripts in `pipeline/` were written with help from Claude (an AI assistant). We're going through them and re-running them ourselves. We'll keep this section updated as the project goes.
