# Where Explanations Disagree

Final project for DS-GA 3001 Visualization for Machine Learning (NYU, Fall 2026).
Team 7: Sheetal Sood and Adarsh Tiwari.

**Live tool:** https://tiwari-adarsh0830.github.io/where-explanations-disagree/ (GitHub Pages, same files as this repo; the greater-than task is at `frontend/?task=gt`).

## What we're doing

Explanation methods for language models don't always agree. Attention maps, logit attribution and gradient methods can point at different attention heads for the same prediction. We want to build a visual analytics tool that shows where they disagree, and use activation patching (knocking out a head or a group of heads and checking what happens to the output) as a check on which explanation holds up.

We're starting with GPT-2 small on the indirect object identification (IOI) task. Wang et al. (2023) already mapped the circuit GPT-2 uses for it, so we have something to compare against.

**Status:** in progress. The proposal was approved in October and the project update went in on Nov 3. Pipeline v3 and the analysis are done, we've rerun patching under five different settings, the second task (greater-than) is in, and we added assumption checks that test why a method disagrees with patching. The D3 tool covers all of it and switches between the two tasks.

## What's in here

| Folder | What it has |
| --- | --- |
| `pipeline/` | Python scripts that compute the explanations and patching effects. Older versions are in `pipeline/archive/` |
| `data/` | `ioi_lists.json`: templates, names, places and objects from Wang et al.'s dataset code. `gt_nouns.txt`: the 120 nouns Hanna et al. used for greater-than (from their code release) |
| `results/` | Pipeline and analysis output (JSON, plus a per-head table from v2). Greater-than output is in `results/gt/` |
| `frontend/` | The D3 tool (one HTML page, see `frontend/README.md`) |
| `tests/` | Headless browser checks of the page against the results (`bash tests/run_all.sh`) |
| `docs/` | Links to our planning docs |
| `index.html`, `.nojekyll` | Let GitHub Pages serve the repo as the live tool: the root page forwards to `frontend/` |

In `pipeline/`, `ioi_check_v3.py` is the current pipeline. It's the same as v2 (120 prompts from the paper's templates) but it keeps every prompt's scores, labels each prompt with its template and name order, and records attention at five query/key positions instead of one. `analysis_v4.py` reads that output plus `attn_scores.json` and produces every number the D3 page and the paper use: agreement with patching for each method with bootstrap intervals, attention at six positions (raw and value-weighted), the attribution patching error analysis, the group knockouts, and the per-order and per-template breakdowns. `patching_variants.py` reruns single-head patching and the group knockouts under five settings (noising vs denoising, three kinds of corruption, all positions vs the final token only) on the same 120 prompts, and `analysis_variants.py` compares them. `analysis_v3.py` is the earlier version behind our Nov 3 update (attention at five positions). `archive/ioi_check_v2.py` produced the numbers in our proposal and `archive/ioi_smoke_test_v1.py` was our first rough try. We kept both for reference.

## Setup

We ran this on Python 3.13 with CPU PyTorch.

```
pip install -r requirements.txt
```

TransformerLens 4.x removed `HookedTransformer`, which the IOI code and most tutorials use, so `requirements.txt` pins it below 4.

## Running it

```
python pipeline/ioi_check_v3.py data/ioi_lists.json results/ioi_v3_results.json
python pipeline/attention_patterns.py data/ioi_lists.json results
python pipeline/analysis_v4.py results/ioi_v3_results_per_prompt.json results/attn_scores.json results/analysis_v4.json
python pipeline/patching_variants.py data/ioi_lists.json results
python pipeline/analysis_variants.py results/patching_variants_per_prompt.json results/ioi_v3_results_per_prompt.json results/attn_scores.json results/analysis_variants.json
python pipeline/gt_check.py data/gt_nouns.txt results/gt
python pipeline/analysis_gt.py results/gt results/analysis_gt.json
python pipeline/attention_map.py data/ioi_lists.json results
python pipeline/analysis_baselines.py results results/analysis_baselines.json
python pipeline/diagnostics.py data/ioi_lists.json results both
python pipeline/analysis_diagnostics.py results
python pipeline/check_interventions.py
```

GPT-2 small downloads from Hugging Face the first time. The pipeline takes about 4.5 minutes on a MacBook CPU (about 19 on a 2 core cloud machine) because patching every head one at a time is slow. The analysis takes a few seconds. `attention_patterns.py` only runs forward passes (under a minute) and saves each head's attention patterns for the D3 tool, plus attention at a sixth position (the token after S1 looking back at S1, where previous-token heads act) and a value-weighted version of attention. v3's averages match v2 exactly, so the proposal numbers still hold. `patching_variants.py` batches the 12 heads of a layer together and takes about half an hour on the cloud machine; it checks that its first setting reproduces v3's patching numbers. `diagnostics.py` runs the extra interventions behind the assumption checks below (about 45 minutes on the cloud machine for both tasks) and checks that its full-strength patching reproduces the v3 and greater-than numbers (largest difference 0.0001).

To check the page itself: `pip install playwright && python -m playwright install chromium`, then `bash tests/run_all.sh` from the repo root. It serves the repo locally, loads every view in a headless browser for both tasks, compares numbers on the page with the result files, and checks for script errors, sideways overflow on a phone-width screen, keyboard selection, shareable links and the guided tour. With `cd tests && npm i axe-core@4` it also runs an automated WCAG AA audit.

## First results (one run, so preliminary)

- GPT-2 small got all 120 prompts right (mean logit difference 4.19).
- Across the 144 heads, attention agreed only weakly with single-head patching (Spearman 0.17). Logit attribution got 0.36 and attribution patching 0.88.
- Attention ranks the S-inhibition heads 23rd to 69th. Patching puts them 2nd to 9th.
- Knocking out the 3 name movers drops the logit difference by only 0.11 on average. That average hides a split though: it drops by 0.67 when the subject comes first (BABA) and goes up by 0.45 when the indirect object comes first (ABBA). Knocking them out together with the 8 backup name movers drops it by 2.66.

So attention misses part of the circuit, and patching one head at a time misses the backup heads. That's why we want group patching in the tool.

Since then we found that attention mostly misses those heads because we only looked at the final token. Scoring each head where it actually looks (for example the S-inhibition heads looking at the second name) raises its agreement with patching from 0.17 to 0.42. Bootstrap intervals and the rest are in `results/analysis_v3.json`.

## Does it depend on how you patch? (Nov 21)

Everything above uses one patching setup. We reran it four other ways and the answer to "which explanation agrees with patching" changes a lot (numbers in `results/analysis_variants.json`):

- The patching settings don't agree with each other that well. Noising and denoising with the same corruption only reach Spearman 0.47 and share 4 of their top 10 heads. Mean ablation is the only one close to the original (0.82, same top 10).
- Final-token attention goes from the worst method (0.17) to one of the better ones (about 0.49) if the reference is denoising or patching the final position only. Under denoising, logit attribution has the same top 10 heads as patching (10 of 10).
- Attribution patching only stands out against the setting it approximates (0.88, and 0.80 for mean ablation). Against the others it's 0.49 to 0.60.
- The name-mover sign flip between BABA and ABBA shows up with three-new-names corruption and mean ablation, but not with the name-flip corruption or denoising, where knocking out the name movers hurts in both orders.

## Second task: greater-than (Nov 21)

To check whether any of this is specific to IOI, `gt_check.py` runs the same five scores on the greater-than task from Hanna et al. (2023): "The war lasted from the year 1732 to the year 17", where GPT-2 should prefer years after 32. We follow their setup (their 120 nouns, centuries 11 to 17, years GPT-2 splits into two tokens, probability difference as the metric, "01" as the corruption) with 194 prompts, two per start year. GPT-2 gets a mean probability difference of 0.83 (they report 0.82). Single-head patching puts all seven of their circuit heads in its top 10, so the setup reproduces. Numbers below are from `results/analysis_gt.json`:

- Here plain attention (final token to the start year) does well at the top: its top 10 contains all seven circuit heads and overlaps patching's top 10 in 8 of 10. Reading attention at the "best" position makes it worse (5 of 10), the opposite of IOI. In this task every circuit head acts from the final token, so there's one natural place to look.
- Logit attribution is the weakest method (Spearman 0.31, ranks 5.5 99th and 5.1 85th), which fits Hanna et al.'s account that those heads work through MLPs 8 to 11, not directly on the output.
- Noising and denoising agree on 9 of their top 10 heads (in IOI it was 4 of 10), and knocking out both head groups is only 1.25 times the two groups separately (4.1 in IOI). Less backup behavior to confuse the two settings.

## Two checks on our own numbers (Nov 21)

**Baselines.** A score that just says "later heads matter more" already agrees with patching. Scoring each head by its layer number gets Spearman 0.22 against our main patching setup in IOI (higher than final-token attention, 0.17), 0.47 against the name-flip setup, and 0.76 against final-position patching, where it beats every method. In greater-than, layer number is negative (-0.20), but "how much of the final token's attention a head spends anywhere except the first token" gets 0.43, as much as our best attention score. Every agreement number should be read against these (`results/analysis_baselines.json`, from `pipeline/analysis_baselines.py`).

**Is "best position" just selection?** `pipeline/attention_map.py` scores attention at every query/key position (IOI positions aligned by role, greater-than all 13 x 13 token pairs) and tests position choices on held-out data (`results/analysis_attn_map.json`). In IOI no single position beats 0.26, our hand-picked six reach 0.52 on held-out templates (0.54 on all), and taking each head's max over every position drops to 0.17, so the gain comes from picking the right positions, not from having more of them. Picking positions from the data does about as well (0.50 held out with six). In greater-than the data-picked pair (final token to the start year and to the first "year") beats our hand-picked five on held-out prompts, 0.58 against 0.44.

## Why do the methods disagree? Assumption checks (Nov 22)

Each method agrees with patching only if one assumption holds for a head. Final-token attention assumes the head does its work at the final token. DLA assumes the effect goes straight to the output. Attribution patching assumes the effect grows linearly with the change. Single-head patching assumes no other head changes this head's effect. `pipeline/diagnostics.py` tests each assumption for every head with its own intervention, under our main patching setting: patching the final position only, patching only the head's direct path to the output (final LayerNorm frozen), patching at half strength, and patching the 276 pairs among the 24 most important heads. `pipeline/analysis_diagnostics.py` turns each into a violation score between 0 (holds) and 1 (breaks) and asks whether the violations explain where each method disagrees with patching (`results/analysis_diagnostics.json`).

The result is mixed, and we report it that way. Every number below also comes with a robustness test: we repeat it under 27 choices of cutoffs (which heads count, top 20/30/40 for either score; what counts as misranked, ranks 1.5x/2x/3x apart; what counts as broken, a violation above 0.3/0.5/0.7) and resample the prompts. The interventions themselves were checked against independent forward passes with TransformerLens hooks (`pipeline/check_interventions.py`: direct-path and pair effects match to 0.0001).

- One check clearly explains disagreements: attribution patching. In IOI it misranks 7 of the heads in either top 30 (ranks at least 2x apart); 4 of them bend away from linear, against 3 of the 27 it gets right (one-sided Fisher p = 0.02). This holds under all 27 cutoff choices. In greater-than attribution patching misranks no important head at all (Spearman 0.96).
- One check points the right way but weakly: final-token attention. In IOI, 15 of the 40 heads it misranks act mainly at earlier tokens, against 0 of the 7 it gets right. The direction holds under all 27 cutoff choices, but few heads count as "right" (7 at the default cutoffs), so it is significant (p < 0.1) in only 8 of them. Greater-than: same direction in 24 of 27, significant in 10. Head 5.5 is the clearest case: all of its effect happens at earlier tokens and none goes directly to the output, so final-token attention ranks it 73rd and DLA 113th while patching ranks it 3rd.
- Two assumptions break for most important heads whether or not the method agrees. Single-head patching's no-partners assumption fails for every important head in both tasks, under every cutoff choice (all 7 heads where noising and denoising agree in IOI, all 17 in greater-than). DLA's direct-path assumption fails for 8 of the 13 heads where DLA agrees with patching in IOI (19 of the 30 it misranks); only for gross misranks (3x apart) does it break clearly more often (17 of 21 against 10 of 22). In greater-than it never separates. So for these two methods, matching patching on a head is weak evidence that the method reads the head correctly. In greater-than, head 9.1 is DLA's top head and patching's top head, but only 17% of its effect goes through the direct path; the rest goes through later layers, as Hanna et al. describe for the MLPs.
- The checks are not specific to their own method. In IOI a head that breaks one assumption usually breaks the others (Spearman between violations and rank gaps is 0.4 to 0.6 almost everywhere in the 4 x 4 table, and still is after controlling for the size of the head's effect), because the middle of the circuit (induction and S-inhibition heads) acts at earlier tokens, through later heads, and with backups all at once.
- Some of final-token attention's agreement in IOI is just depth: 0.17, and 0.08 (95% interval 0.03 to 0.16) once layer is held out. Best-position attention (0.54 to 0.56) and attribution patching (0.88) keep their agreement with layer held out.

## Main references

- Wang et al. 2023, *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*
- Hanna, Liu & Variengien 2023, *How does GPT-2 compute greater-than?* (NeurIPS)
- Vig 2019, *A Multiscale Visualization of Attention in the Transformer Model* (BertViz)
- Silva et al. 2025, *Visagreement: Visualizing and Exploring Explanations (Dis)Agreement* (IEEE VIS)
- Vaidyanathan et al. 2026, *The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching*
- Zhang & Nanda 2024, *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods* (ICLR)
- Heimersheim & Nanda 2024, *How to use and interpret activation patching*

## AI use

Following the course AI policy: the scripts in `pipeline/` and the D3 page in `frontend/` were written with help from Claude (an AI assistant). We run everything ourselves and check the numbers against each other. We'll keep this section updated as the project goes.
