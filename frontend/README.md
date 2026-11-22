# Frontend

The D3 tool. It's one HTML page (`index.html`) that reads `../results/ioi_v3_results_per_prompt.json` and works out everything else in the browser.

To run it, serve the repo root and open the frontend folder:

```
python3 -m http.server 8000
```

Then go to http://localhost:8000/frontend/. Opening the file directly doesn't work because browsers block loading the JSON from a `file://` page. D3 comes from a CDN, so you need internet.

What's there so far:

- **Grids:** one layer by head grid per method (attention at the final token, attention at its best position, DLA, attribution patching, activation patching), colored by rank or by signed value.
- **Rank scatter:** any method's rank against patching rank, so heads far from the diagonal are the ones the method gets wrong.
- **Group knockouts:** knocking out the name movers, the backups or both at once, next to what adding up single heads would predict, split by name order.
- **Selected head panel and table view:** click a head anywhere and it's selected in every view.

The name order filter (pooled, BABA, ABBA) changes every view at once.
