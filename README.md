# Organ-specific proteomic clocks

Run from this directory, in order:

```bash
python 01_train_discovery_clocks.py
python 02_score_independent_proteomes.py
python 03_compare_clock_proteins.py
python 04_score_bulk_rna.py
python 05_score_spatial_rna.py
```

`01` reads the Scifo mzTab files and trains the eight organ clocks. `02` scores Takasugi, Lu, Burns, and Watanabe with those weights. `03` tests age-differential proteins, compares the core proteins across organs, and retrains on the proteins detected in all eight organs. `04` scores Takasugi and Shavlakadze RNA. `05` scores the Visium sections, niches, and hotspots.

The scripts look upward for a directory that contains `Dataset/`, and write `results/` there. If this folder is moved, set `PROTEOMIC_CLOCK_ROOT` to that directory.

Inputs:

- `Dataset/Scifo/`
- `Dataset/Takasugi/`
- `Dataset/Lu_intervention/`
- `Dataset/Burns/`
- `Dataset/Orwell_intervention/`
- `Dataset/GSE314817/`
- `Dataset/Spatial_Transcriptome/GSE283283_aging/`

`04` and `05` also need `results/human_clock_overlap/mouse_stable_with_genes.csv`, with columns `organ`, `accession`, `gene`, `selection_frequency`, `mean_coef`, and `description`.

Dependencies are listed in `requirements.txt`.
