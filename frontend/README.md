# Frontend

The D3 tool. It's one HTML page (`index.html`) that reads `../results/ioi_v3_results_per_prompt.json`, `../results/attn_scores.json` and `../data/ioi_lists.json`, and works out everything else in the browser. When you select a head it also loads that head's attention patterns from `../results/attn/`.

To run it, serve the repo root and open the frontend folder:

```
python3 -m http.server 8000
```

Then go to http://localhost:8000/frontend/. Opening the file directly doesn't work because browsers block loading the JSON from a `file://` page. D3 comes from a CDN, so you need internet.

What's there so far:

- **Grids:** one layer by head grid per method (attention at the final token, attention at its best position, DLA, attribution patching, activation patching), colored by rank or by signed value.
- **Rank scatter:** any method's rank against patching rank, so heads far from the diagonal are the ones the method gets wrong. Value-weighted attention is also available here.
- **Where a head looks:** the selected head's full token by token attention for any template and name order, raw or value-weighted, with the six scored positions outlined.
- **Group knockouts:** knocking out the name movers, the backups or both at once, next to what adding up single heads would predict, split by name order.
- **Per template:** clean logit difference, agreement with patching, the name-mover knockout and the selected head's effect for each of the 15 templates, BABA next to ABBA.
- **Selected head panel and table view:** click a head anywhere and it's selected in every view.

The name order filter (pooled, BABA, ABBA) changes every view at once.
