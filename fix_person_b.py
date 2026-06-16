import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('d:/faculta/DM/Data-mining-project/airbnb_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

# ── Cell 136: Replace cluster label mapping (luxury/mid-range/budget → 2-cluster labels) ──
cells[136]['source'] = [
    "# ------------------------------------------------------------------ #\n",
    "# Assign human-readable labels based on cluster profile              #\n",
    "# ------------------------------------------------------------------ #\n",
    "\n",
    "# With best_k = 2 the model finds two broad market groups.\n",
    "# We rank by mean price so Cluster 0 = lower-price group, Cluster 1 = higher-price group.\n",
    "cluster_name_map = {\n",
    "    0: 'Budget / compact listings',\n",
    "    1: 'Larger / premium listings',\n",
    "}\n",
    "\n",
    "# If the price-rank ordering differs from 0/1, remap automatically\n",
    "price_rank = cluster_profile['price'].rank().astype(int)\n",
    "sorted_clusters = price_rank.sort_values().index.tolist()  # [low_price_cluster, high_price_cluster]\n",
    "cluster_labels = {\n",
    "    sorted_clusters[0]: 'Budget / compact listings',\n",
    "    sorted_clusters[1]: 'Larger / premium listings',\n",
    "}\n",
    "\n",
    "listings_clust['cluster_label'] = listings_clust['cluster'].map(cluster_labels)\n",
    "print('Cluster label mapping:')\n",
    "for k_id, lbl in sorted(cluster_labels.items()):\n",
    "    n = (listings_clust['cluster'] == k_id).sum()\n",
    "    print(f'  Cluster {k_id} \u2192 {lbl}  (n={n:,})')\n",
    "\n",
    "# Updated cluster profile table grouped by readable label\n",
    "cluster_profile_labelled = (\n",
    "    listings_clust\n",
    "    .groupby('cluster_label')[cluster_features]\n",
    "    .mean()\n",
    "    .round(2)\n",
    ")\n",
    "print('\\nCluster profile (grouped by label):')\n",
    "display(cluster_profile_labelled)\n"
]

# ── Cell 139: Replace markdown interpretation (luxury wording → 2-cluster wording) ──
cells[139]['source'] = [
    "> The final K-Means model selected **2 clusters** based on the silhouette score.  \n",
    "> This means the model found two broad Airbnb market groups rather than detailed categories like budget, mid-range, and luxury.  \n",
    "> **Cluster 0 — Budget / compact listings:** lower prices, lower guest capacity, fewer bedrooms, and fewer amenities.  \n",
    "> **Cluster 1 — Larger / premium listings:** higher prices, more bedrooms, higher guest capacity, and more amenities.  \n",
    "> Because there are only two clusters, it is more accurate to call this group *larger / premium* rather than *luxury* — the model did **not** find a separate luxury segment distinct from the broader premium group.\n"
]

# ── Cell 144: Fix plt.cm.get_cmap → mpl.colormaps (DBSCAN plot) ──
new_cell_144 = []
for line in cells[144]['source']:
    if "palette_db = plt.cm.get_cmap('tab10', max(n_db_clusters, 1))" in line:
        new_cell_144.append("import matplotlib as mpl\n")
        new_cell_144.append("palette_db = mpl.colormaps['tab10'].resampled(max(n_db_clusters, 1))\n")
    else:
        new_cell_144.append(line)
cells[144]['source'] = new_cell_144

# ── Cell 146: Update comparison table footnote (remove luxury mention) ──
new_cell_146 = []
for line in cells[146]['source']:
    if 'Budget / Mid-range / Luxury etc.' in line:
        new_cell_146.append(line.replace(
            'Budget / Mid-range / Luxury etc.',
            'Budget / compact listings or Larger / premium listings'
        ))
    else:
        new_cell_146.append(line)
cells[146]['source'] = new_cell_146

with open('d:/faculta/DM/Data-mining-project/airbnb_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done! Cells 136, 139, 144, 146 updated.")
