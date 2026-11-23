# Where Explanations Disagree

Final project for DS-GA 3001 Visualization for Machine Learning (NYU, Fall 2026).
Team 7: Sheetal Sood and Adarsh Tiwari.

**Live tool:** https://tiwari-adarsh0830.github.io/where-explanations-disagree/ (GitHub Pages, same files as this repo; the greater-than task is at `frontend/?task=gt`).

## What we're doing

Explanation methods for language models don't always agree. Attention maps, logit attribution and gradient methods can point at different attention heads for the same prediction. We want to build a visual analytics tool that shows where they disagree, and use activation patching (knocking out a head or a group of heads and checking what happens to the output) as a check on which explanation holds up.

We work on GPT-2 small with two tasks whose circuits are already mapped: indirect object identification (IOI; Wang et al., 2023) and greater-than (Hanna et al., 2023), so we have something to compare against. The main checks were then repeated on GPT-2 medium (both tasks) and Pythia-410M (IOI), with every rule fixed before those runs.

**Status:** the tool and the analysis are done; the presentation is Dec 8 and the paper is due Dec 14. The proposal was approved in October and the project update went in on Nov 3. Since then: patching under five settings, the greater-than task, depth baselines, five assumption checks that test why a method disagrees with patching, validity checks (split-half, multiple testing, an independent re-computation), and the same checks on GPT-2 medium and Pythia-410M. The D3 tool covers all of it and switches between the two tasks.

## What's in here

| Folder | What it has |
| --- | --- |
| `pipeline/` | Python scripts that compute the explanations and patching effects. Older versions are in `pipeline/archive/` |
| `data/` | `ioi_lists.json`: templates, names, places and objects from Wang et al.'s dataset code (MIT License). `gt_nouns.txt`: the 120 nouns Hanna et al. used for greater-than (from their code release, which has no license file). Sources and license text: `data/NOTICE.md` |
| `results/` | Pipeline and analysis output (JSON, plus a per-head table from v2). Greater-than output is in `results/gt/` |
| `frontend/` | The D3 tool (one HTML page, see `frontend/README.md`) |
| `tests/` | Headless browser checks of the page against the results (`bash tests/run_all.sh`) |
| `docs/` | Links to our planning docs |
| `index.html`, `.nojekyll` | Let GitHub Pages serve the repo as the live tool: the root page forwards to `frontend/` |

In `pipeline/`, `ioi_check_v3.py` is the current pipeline. It's the same as v2 (120 prompts from the paper's templates) but it keeps every prompt's scores, labels each prompt with its template and name order, and records attention at five query/key positions instead of one. `analysis_v4.py` reads that output plus `attn_scores.json` and produces the core agreement numbers: agreement with patching for each method with bootstrap intervals, attention at six positions (raw and value-weighted), the attribution patching error analysis, the group knockouts, and the per-order and per-template breakdowns. `patching_variants.py` reruns single-head patching and the group knockouts under five settings (noising vs denoising, three kinds of corruption, all positions vs the final token only) on the same 120 prompts, and `analysis_variants.py` compares them. `analysis_v3.py` is the earlier version behind our Nov 3 update (attention at five positions). `archive/ioi_check_v2.py` produced the numbers in our proposal and `archive/ioi_smoke_test_v1.py` was our first rough try. We kept both for reference.

## Setup

We ran this on Python 3.13 with CPU PyTorch.

```
pip install -r requirements.txt
```

On Linux, install the CPU build of torch first (`pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`), or pip pulls the much larger CUDA build.

TransformerLens 4.x removed `HookedTransformer`, which the IOI code and most tutorials use, so it has to stay on 3.x. `requirements.txt` pins the exact versions behind the committed results (torch 2.15.0, TransformerLens 3.9.0, transformers 5.22.1, numpy 2.5.4, scipy 1.18.1). Checked Nov 22 from a fresh clone in a fresh environment: every analysis script reproduces the committed result files exactly, `diagnostics.py` reproduces its numbers exactly on a test group, and `attention_patterns.py` matches to the saved rounding. Checked again Nov 23 on a different processor (Intel instead of the MacBook's ARM): numpy's default sort orders tied values differently there, which swapped two tied entries in a few top-10 lists (no count, p-value or verdict changed), so every analysis sort is now stable and the result files were regenerated with it; all of them, including the GPT-2 medium and Pythia analyses, now come out identical on both.

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
python pipeline/diagnostics_values.py data/ioi_lists.json results both
python pipeline/analysis_diagnostics.py results
python pipeline/analysis_validity.py results      # after analysis_diagnostics.py: split-half and multiple-testing checks
python pipeline/check_interventions.py
# another model size (all scores and checks in one script; checked against GPT-2 small first)
python pipeline/model_check.py gpt2-small /tmp/mc_small both 2 && python pipeline/check_model_check.py /tmp/mc_small results --analysis
python pipeline/model_check.py gpt2-medium results/gpt2-medium
python pipeline/analysis_model.py results/gpt2-medium
python pipeline/model_check.py pythia-410m results/pythia-410m ioi && python pipeline/analysis_model.py results/pythia-410m
```

GPT-2 small downloads from Hugging Face the first time. The pipeline takes about 4.5 minutes on a MacBook CPU (about 19 on a 2 core cloud machine) because patching every head one at a time is slow. The analysis takes a few seconds. `attention_patterns.py` only runs forward passes (under a minute) and saves each head's attention patterns for the D3 tool, plus attention at a sixth position (the token after S1 looking back at S1, where previous-token heads act) and a value-weighted version of attention. v3's averages match v2 exactly, so the proposal numbers still hold. `patching_variants.py` batches the 12 heads of a layer together and takes about half an hour on the cloud machine; it checks that its first setting reproduces v3's patching numbers. `diagnostics.py` and `diagnostics_values.py` run the extra interventions behind the assumption checks below (about 45 and 25 minutes on the cloud machine for both tasks) and checks that its full-strength patching reproduces the v3 and greater-than numbers (largest difference 0.0001).

`model_check.py` recomputes every score and check (final-token attention, DLA, attribution patching, noising, half-strength, final-position, denoising, direct-path, value patching at the target, and the 276 pairs) for any GPT-2 size or Pythia model in one script, with the same prompts and corruptions; all GPT-2 sizes share one tokenizer, so the prompts are the same tokens. For another tokenizer it keeps only one-token names (IOI) and only start years that split into century and two digits (greater-than), and it prepends a start-of-text token for every model. Run on GPT-2 small it matches the committed files to the saved rounding (`check_model_check.py`; largest difference 1e-4). `analysis_model.py` runs the contrast test, split-half and Holm correction on its output with the definitions loaded from `analysis_diagnostics.py` and `analysis_validity.py`; fed GPT-2 small's committed numbers it reproduces `analysis_diagnostics.json` and the corrected p-values exactly (`check_model_check.py --analysis`). GPT-2 medium (24 layers x 16 heads) takes about 5 minutes per IOI batch of 4 prompts on the cloud machine; the full medium run took 82 minutes on a MacBook and Pythia-410M's IOI run 53 minutes.

To reproduce a figure: open the page (live link above, or `python3 -m http.server` from the repo root and `/frontend/`), set the view (the address bar keeps the head, method, patching setting and filters), and press **Save SVG** on the card. Each file records the link it came from in its `<desc>` element. For a PDF, `pip install cairosvg` and `cairosvg figure.svg -o figure.pdf`.

To check the page itself: `pip install playwright && python -m playwright install chromium`, then `bash tests/run_all.sh` from the repo root. It serves the repo locally, loads every view in a headless browser for both tasks, compares numbers on the page with the result files, and checks for script errors, sideways overflow on a phone-width screen, keyboard selection, shareable links, the guided tour, the SVG export of every card, the three-model table (every cell recomputed from the result files), and the fixes from a full review of the page (link targets clear the toolbar, Unpin keeps the view, each wide table scrolls on its own on a phone). With `cd tests && npm i axe-core@4` it also runs an automated WCAG AA audit.

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

Each method agrees with patching only if its assumptions hold for a head. Final-token attention assumes the head does its work at the final token, and that what it reads at the target token (IO, or YY in greater-than) carries its effect. DLA assumes the effect goes straight to the output. Attribution patching assumes the effect grows linearly with the change. Single-head patching assumes no other head changes this head's effect. Each assumption gets its own intervention, under our main patching setting:

- `pipeline/diagnostics.py`: patching the final position only; patching only the head's direct path to the output (final LayerNorm frozen); patching at half strength; patching the 276 pairs among the 24 most important heads.
- `pipeline/diagnostics_values.py`: patching only the head's value vectors at one source token (IOI: IO, S1, the token after S1, S2, the final token; greater-than: YY and the final token), which says what the head reads.

`pipeline/analysis_diagnostics.py` turns each into a violation score between 0 (holds) and 1 (breaks) and asks whether the violations explain where each method disagrees with patching (`results/analysis_diagnostics.json`). It compares heads a method misranks (ranks at least 2x apart) with heads it gets right (within 2x), among heads in either top 30 whose patching effect is at least 2% of the largest; below that the violation is a ratio of two near-zero numbers, and the page already calls those heads too small to judge. Every result is repeated under 27 cutoff choices (top 20/30/40; 1.5x/2x/3x apart; violation above 0.3/0.5/0.7) and with resampled prompts, and the unfiltered counts are kept in the output too. All four kinds of intervention were recomputed a second way with plain TransformerLens hooks and full forward passes (`pipeline/check_interventions.py`: direct-path effects, pair effects, half-strength patching through each head's output, and value patching written out through the clean attention pattern); they match to 0.0001 in both tasks.

The result is mixed, and we report it that way:

- One check clearly separates the heads a method gets wrong from the ones it gets right: attribution patching's linearity. In IOI it misranks 7 important heads; 4 of them bend away from linear, against 3 of the 27 it gets right (one-sided Fisher p = 0.02; same direction under all 27 cutoff choices and in all 400 split-half tests; p = 0.034 after correcting for the number of tests, see below). In greater-than it misranks none (Spearman 0.96).
- One check points the right way every time but is not significant once the number of tests is taken into account: final-token attention's "acts at the final token". In IOI 13 of the 26 heads it misranks act mainly at earlier tokens, against 0 of the 7 it gets right (uncorrected p = 0.018; same direction under all 27 cutoff choices and in all 400 split-half tests, but p = 0.10 after correction, because only 7 heads count as right). In greater-than 6 of 12 against 1 of 9 (uncorrected p = 0.08, corrected 0.68). We read this as consistent, not established. Head 5.5 is the clearest case: all of its effect happens at earlier tokens and none goes directly to the output, so final-token attention ranks it 73rd and DLA 113th while patching ranks it 3rd.
- In GPT-2 small, three assumptions break for most important heads whether or not the method agrees, so agreement with patching proves little there (on GPT-2 medium and Pythia-410M two of them behave differently, see below). Single-head patching's no-partners assumption fails for every important head in both tasks. DLA's direct-path assumption fails for 8 of the 13 heads where DLA agrees with patching in IOI (15 of the 22 it misranks); only gross misranks, 3x apart, break it clearly more often (13 of 14 against 10 of 21). In greater-than head 9.1 is DLA's and patching's top head, yet only 17% of its effect goes through the direct path, the rest through later layers, as Hanna et al. describe for the MLPs. And "what it reads at IO carries its effect" fails for 5 of the 7 heads final-token attention gets right in IOI: only the name movers carry their effect through IO's values. Value patching does say what the other heads read: the S-inhibition heads read S2 (8.6: +1.55 of its +1.53 effect), the duplicate and previous-token heads read S1, and the induction heads (5.5, 6.9) are carried by no single token's values, so they work through where they look rather than what they read.
- The checks are not specific to their own method. In IOI a head that breaks one of the first four assumptions (final token, direct, linear, no partners) usually breaks the others too (Spearman between violations and rank gaps is about 0.4 to 0.7 in most cells of that 4 x 4 table, and still is after controlling for the size of the head's effect; the exceptions are attribution patching against the final-token and direct checks), because the middle of the circuit (induction and S-inhibition heads) acts at earlier tokens, through later heads, and with backups all at once.
- Some of final-token attention's agreement in IOI is just depth: 0.17, and 0.08 (95% interval 0.03 to 0.16) once layer is held out. Best-position attention (0.54 to 0.56) and attribution patching (0.88) keep their agreement with layer held out.

**Are these results real?** `pipeline/analysis_validity.py` (`results/analysis_validity.json`) runs two checks on the test above.
- Shared noise: the rank gap and the violation both come from the same patching runs, so noise in one prompt set could move both together. Split-half version: decide which heads are misranked on one half of the prompts (IOI split by template) and measure the violations on the other half, 200 splits both ways round. The results barely change: attribution patching 400 of 400 in the same direction (p < 0.1 in 363), final-token attention in IOI 400 of 400 (370), in greater-than 400 of 400 (216). The matching table also comes out the same (for example attribution patching against its own check: 0.53 split-half, 0.55 on all prompts), so the non-specificity is not shared noise either.
- Multiple tests: we test several checks in two tasks. Holm correction over the eight tests planned before the results (four checks, two tasks), with a one-sided Mann-Whitney test on the violation scores (it keeps their size, unlike the 0.5 cutoff): attribution patching in IOI p = 0.034 (permutation test 0.011); final-token attention in IOI 0.10 (0.08), greater-than 0.68 (0.12); everything else above 0.18. Adding the later "reads the target" check to the family (ten tests) leaves attribution patching at 0.043. So one result survives every check we ran; final-token attention's is consistent but underpowered.

**Does it hold on GPT-2 medium?** (Nov 22; `results/gpt2-medium/`, from `pipeline/model_check.py` and `pipeline/analysis_model.py`.) GPT-2 medium has 24 layers of 16 heads. Same prompts, corruptions and metrics, and every definition (2% rule, top-30 pool, 2x cutoff, 0.5 threshold, Holm family) fixed before the run, so this is a test on a model the rules were not tuned on. Medium does both tasks (IOI logit difference 4.19 clean, -0.16 corrupted; greater-than 0.87 and -0.25). Corrected p is the Holm-adjusted one-sided Mann-Whitney test within medium's own family of planned tests.
- Single-head patching's no-partners assumption breaks for every important head again, in both tasks (IOI 19 of 19 misranked and 5 of 5 right; greater-than 7 of 7 and 17 of 17). This is the one result that holds everywhere we looked.
- Attribution patching's linearity points the same way but the test has almost nothing to work with: attribution patching agrees with patching at Spearman 0.98 in IOI and misranks only 2 heads. One of them bends strongly (7.2, violation 0.99), the other does not (11.6, 0.38), against 1 of the 31 heads it gets right. Corrected p = 0.038, but that rests on one head, so we count it as consistent, not as a second confirmation. In greater-than it misranks none (Spearman 1.00), as in small.
- Final-token attention's "acts at the final token" points the right way again in IOI (8 of 34 misranked against 0 of 6 right; mean violation 0.24 against 0.01). The Mann-Whitney test is significant after correction (0.013) but the count test is not (Fisher 0.24), and only 6 heads count as right. Still consistent, still small groups. Greater-than: 2 of 12 against 0 of 18, corrected 0.82 and Fisher 0.15, so it only leans the same way; in small it reached Fisher 0.08.
- Two assumptions that broke almost everywhere in small separate cleanly in medium IOI. DLA's direct-path assumption breaks for 25 of the 32 heads DLA misranks and for none of the 9 it gets right (corrected p = 0.0007; same direction under all 27 cutoff choices and in all 400 split-half tests); 6 of the 9 are in layers 20 to 23, the others are 12.3, 15.4 and 17.4. "What it reads at IO carries its effect" breaks for 27 of 34 misranked against 1 of 6 right (corrected p = 0.0009 in the family of ten). Both hold without the 2% rule too (29 of 36 against 0 of 9; 37 of 45 against 1 of 6). In greater-than neither separates (DLA 22 of 24 against 7 of 7). We do not know why medium differs from small here; it means "agreement with patching proves little for DLA" is a GPT-2 small IOI result, not a general one.

**Does it hold on a different model family? Pythia-410M** (Nov 23; `results/pythia-410m/`, IOI only.) Pythia-410M (EleutherAI) has 24 layers of 16 heads like GPT-2 medium, but a different tokenizer, rotary position embeddings, and attention and MLP in parallel. Same templates, corruption, metric and fixed rules as medium. Its tokenizer splits 11 of the 99 names into two tokens, so its 120 prompts use the same templates and seed with the other 88 names. Greater-than was not run: Pythia splits 48 of the 194 start years differently (" 160" + "4" instead of " 16" + "04"), and on the other 146 only about a quarter of its next-token probability falls on two-digit years (clean minus corrupted 0.30, against 1.34 in small), so the metric does not capture its answer; we decided this from forward passes before any full patching run. Pythia does IOI (logit difference 3.98 clean, -0.03 corrupted; the IO name wins in all 120 prompts). The Holm correction counts the four greater-than tests as not significant, so the family is the same eight (ten with target) as in the two-task runs.
- Agreement with patching: attribution patching 0.96, DLA 0.46, final-token attention 0.35, layer number alone 0.04.
- No partners breaks again: 20 of 20 misranked and 3 of 4 right. It holds in all three models and both tasks we ran.
- Attribution patching's linearity separates again: 4 of 8 misranked heads bend against 3 of 27 right (Fisher 0.034, corrected 0.004; same direction under all 27 cutoff choices, p < 0.1 in all 27; split-half 400 of 400, p < 0.1 in 398). With small, this is the result that holds across two model families; in medium it rested on 2 misranked heads.
- DLA's direct-path assumption breaks for 23 of 34 misranked against 2 of 9 right (Fisher 0.019, corrected 0.062; 27 of 27 cutoffs, p < 0.1 in 25; split-half 400 of 400, p < 0.1 in 398). "What it reads at IO carries its effect": 26 of 35 against 1 of 7 (Fisher 0.005, corrected 0.051 in the family of ten; 27 of 27 cutoffs, p < 0.1 in 18; split-half 398 of 400, p < 0.1 in 312). Both point the right way, as in medium and unlike small, though neither clears 0.05 after correction here.
- Final-token attention's "acts at the final token" only leans the right way: 10 of 35 against 0 of 7 (Fisher 0.12, corrected 1.00; same direction in 27 of 27 cutoffs and 398 of 400 split-half tests, but p < 0.1 in only 13 and 120 of them).
- Across the three models: no partners breaks everywhere; linearity separates in small and Pythia; the direct and reads-IO assumptions track the methods' mistakes in medium and Pythia but not in small, so small is the odd one out there; "acts at the final token" is consistent in direction but the weakest result.
- The checks still move together: in medium IOI the direct-path violation follows DLA's rank gap (0.68) exactly as much as denoising's (0.68), and the position violation follows DLA's gap (0.54) more than attention's own (0.45).

The page shows this comparison as a table under the contrast table on the assumption-check card, with one verdict per check and model under the same rule (checked in order: breaks even where the method agrees, separates when both the count test and the corrected test are below 0.05, points the right way when misranked heads break more often and one test is below 0.1, leans the right way when no test is). So across two model sizes: no-partners breaks everywhere; linearity and "acts at the final token" point the right way in every IOI test but rest on few heads; whether DLA and "reads IO" fail only where those methods disagree depends on the model; nothing separates in greater-than in either model.

## Main references

- Wang et al. 2023, *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*
- Hanna, Liu & Variengien 2023, *How does GPT-2 compute greater-than?* (NeurIPS)
- Vig 2019, *A Multiscale Visualization of Attention in the Transformer Model* (BertViz)
- Silva et al. 2025, *Visagreement: Visualizing and Exploring Explanations (Dis)Agreement* (IEEE VIS)
- Vaidyanathan et al. 2026, *The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching*
- Zhang & Nanda 2024, *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods* (ICLR)
- Heimersheim & Nanda 2024, *How to use and interpret activation patching*
- Zhang & Wang 2026, *When Attribution Patching Lies: Diagnosis and a Second-Order Correction* (NeurIPS)
- Kramár et al. 2024, *AtP\*: An efficient and scalable method for localizing LLM behaviour to components*; Syed, Rager & Conmy 2023, *Attribution Patching Outperforms Automated Circuit Discovery*
- McGrath et al. 2023, *The Hydra Effect: Emergent Self-repair in Language Model Computations*; Rushing & Nanda 2024, *Explorations of Self-Repair in Language Models* (ICML)
- *An Adversarial Example for Direct Logit Attribution: Memory Management in gelu-4l* (BlackboxNLP 2024, arXiv 2310.07325)
- Hanna, Pezzelle & Belinkov 2024, *Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms* (COLM)
- Neely et al. 2022, *A Song of (Dis)agreement: Evaluating the Evaluation of Explainable Artificial Intelligence in Natural Language Processing*
- Kadir & Sonntag 2026, *What Do Visual-Head Discovery Scores Actually Find?* (arXiv 2610.13111)
- HeadVis: An Interactive Tool For Investigating Attention Heads (transformer-circuits.pub, 2026)

How our results sit against these: each single diagnostic we run has precedent (attribution patching's nonlinearity, interaction effects in patching, DLA missing indirect and compensated effects, and the general point that agreement between methods is not evidence of faithfulness). What we add is narrower: for each method, on the heads where it agrees with patching, we test with an independent intervention whether the assumption it relies on holds there, and compare those heads with the ones it misranks. We also put all of it in one tool.

## License

Our code (pipeline, page, tests) and results are under the MIT License, see `LICENSE`. Files we did not write keep their own terms: the IOI lists (MIT) and the greater-than nouns are described in `data/NOTICE.md`, and `tests/d3.min.js` is d3 (ISC License, text in `tests/LICENSE-d3.txt`). The page itself loads d3 from a CDN.

## AI use

Following the course AI policy: the scripts in `pipeline/` and the D3 page in `frontend/` were written with help from Claude (an AI assistant). We run everything ourselves and check the numbers against each other. We'll keep this section updated as the project goes.
