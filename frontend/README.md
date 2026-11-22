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
- **Selected head panel and table view:** click a head anywhere and it's selected in every view.

The name order filter (pooled, BABA, ABBA, IOI only) and the patching setting change every view at once.
