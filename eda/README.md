# EDA

This folder is for lightweight exploratory analysis.

Ideas
- Dataset inventory: counts per class and split; class imbalance ratios.
- Image stats: resolution distribution, aspect ratios, file sizes.
- Color stats: per-class mean/std (RGB), histograms.
- Feature quality: PCA/TSNE plots for feature sets (set1/set2/set4).
- Nearest neighbors: sample-level similarity to spot duplicates/outliers.
- Binary view: healthy vs non-healthy counts and example grids.
- Cluster sanity checks: purity/entropy vs features; inspect cluster sample grids.
- Error audit: highlight images assigned to wrong cluster in best config.

Outputs
- Suggested output paths: output/eda/*.png and output/eda/*.csv.
- Healthy label assumption: Healthy_Nail is the only healthy class (update if needed).

Scripts
- `eda/dataset_inventory.py`: per-class counts and healthy vs non-healthy counts per split.
- `eda/image_stats.py`: resolution/aspect ratio histograms + per-class file size summary.
- `eda/color_stats.py`: mean images + HSV histograms/means (per class and overall).
- `eda/color_stats.py` also outputs H-only histograms, a Euclidean distance heatmap, and an overlay line chart.
- `eda/hog_stats.py`: mean HOG descriptors per class, Euclidean distance matrix, and overlay line chart.
- `eda/lbp_stats.py`: mean LBP histograms per class, Euclidean distance matrix, and overlay line chart.
