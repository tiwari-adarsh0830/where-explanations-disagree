# Frontend

The D3 tool. It's one HTML page (`index.html`) that reads `../results/ioi_v3_results_per_prompt.json`, `../results/attn_scores.json`, `../results/patching_variants_per_prompt.json` (optional) and `../data/ioi_lists.json`, and works out everything else in the browser. When you select a head it also loads that head's attention patterns from `../results/attn/`.

It covers two tasks. The Task switch at the top (or `?task=gt` in the address) loads the greater-than data from `../results/gt/` instead, and every view relabels itself: start years instead of templates, the year heads from Hanna et al. instead of the IOI circuit, and no name-order filter. Switching reloads the page.

The live version is on GitHub Pages: https://tiwari-adarsh0830.github.io/where-explanations-disagree/ (it forwards to this folder).

To run it, serve the repo root and open the frontend folder:

```
python3 -m http.server 8000
```

Then go to http://localhost:8000/frontend/ (or http://localhost:8000/frontend/?task=gt for greater-than). Opening the file directly doesn't work because browsers block loading the JSON from a `file://` page. D3 comes from a CDN, so you need internet.

What's there so far:

- **Grids:** one layer by head grid per method (attention at the final token, attention at its best position, DLA, attribution patching, activation patching), colored by rank or by signed value.
- **Rank scatter:** any method's rank against patching rank, so heads far from the diagonal are the ones the method gets wrong. Value-weighted attention and two baselines (layer number, head activity) are also available here, and the readout always shows the baselines for scale.
- **Where a head looks:** the selected head's full token by token attention for any template and name order, raw or value-weighted, with the six scored positions outlined.
- **Where attention should be read:** a map of how well attention at every query/key position agrees with patching (IOI positions by role, greater-than every token pair). Click a cell to put that position in the rank scatter. Reads `../results/attn_map.json` (or `../results/gt/attn_map.json`).
- **How you patch:** a matrix of how well each method and each patching setting agrees with each patching setting. Click a column, or use the Patching setting menu in the toolbar, to make that setting the reference for every other view.
- **Group knockouts:** knocking out the name movers, the backups or both at once, next to what adding up single heads would predict, split by name order.
- **Per template:** clean logit difference, agreement with patching, the name-mover knockout and the selected head's effect for each of the 15 templates, BABA next to ABBA.
- **Selected head panel and table view:** click a head anywhere and it's selected in every view. You can also type a head (for example 9.9) in the Head box, or focus a grid and move with the arrow keys.
- **Why the methods disagree about this head:** for the selected head, the assumption behind each method, the intervention that tests it, and whether it holds. Final-token attention gets two rows: where the head acts (final-position patching) and what it reads (value patching at the target token, from `pipeline/diagnostics_values.py`), which also names the token that does carry the effect (for example S2 for the S-inhibition heads). Reads `../results/diagnostics_summary.json` (or `../results/gt/diagnostics_summary.json`) from `pipeline/analysis_diagnostics.py`. A "Copy this report" button puts the panel's text on the clipboard.
- **Where each assumption breaks:** one grid per assumption, a scatter of violation against rank gap, the 4 x 4 matching table, and a table of how often each assumption breaks for heads a method misranks against heads it gets right.
- **Which heads change each other's effect:** the pairwise interaction matrix for the 24 heads with the largest patching effects. Clicking a cell pins that pair below the matrix: each head alone, the sum, and the two patched together.
- **Does it hold prompt by prompt:** agreement with patching on each prompt, and the selected head's score against its patching effect on each prompt, with the prompts where it matters most and least.
- **Rank intervals:** a checkbox in the scatter draws each head's 95% bootstrap interval on both ranks.
- **Notes on every view:** each card has a "What this view shows and leaves out" section (unit, evidence, scope, whether it is causal, what it leaves out, where it can mislead).
- **Start here and the guided tour:** a short intro with the main findings (computed from the data on load) and a step-by-step tour for each task.
- **Shareable links:** the address bar keeps the selected head, method, patching setting and filters, so a link reopens the same view (for example `?head=5.5&ref=denoise_abc`).

The name order filter (pooled, BABA, ABBA, IOI only) and the patching setting change every view at once, except the assumption checks, which always use the main patching setting (noising with the task's main corruption) because that is what `pipeline/diagnostics.py` measured.

`bash tests/run_all.sh` from the repo root runs the headless checks of every view (see the main README).
