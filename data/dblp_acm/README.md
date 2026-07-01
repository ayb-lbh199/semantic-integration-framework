# DBLP-ACM benchmark

The DBLP-ACM entity resolution benchmark is publicly available and widely used.
It is not redistributed here. Download it from its original source (for example
the Database Group Leipzig benchmark collection) and place the source CSV files
in this directory.

Expected files:
- `DBLP.csv`
- `ACM.csv`
- `DBLP-ACM_perfectMapping.csv` (ground-truth matches)

Once present, the resolution method in `src/module2_entity_resolution.py` can
be run on these records to demonstrate the pipeline on a public benchmark.
