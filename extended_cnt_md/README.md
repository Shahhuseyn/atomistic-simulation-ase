# Extended: Molecular Dynamics of a (5,5) Carbon Nanotube under Tension

An extended version of the practical that goes beyond the Cu/EMT study in the main folder.
It covers Section 4 of the practical: uniaxial stretching of a (5,5) armchair carbon nanotube
with the AIREBO (Brenner-family) potential, run through LAMMPS from ASE.

📄 [Extended report](report_extended.pdf) | 💻 [Extended notebook](atomistic_simulation_extended.ipynb)

| Task | Content | Script | Output |
|---|---|---|---|
| 4.1 | CNT classification (zigzag, armchair, chiral) | — | in the notebook |
| 4.2 | NVE stretching: time step, strain increment and relaxation time sweeps; critical deformation rate | `scripts/task42_cnt.py` | `results/out42/` |
| 4.3 | Kinetic, potential and total energy vs. strain | `scripts/task43_plot.py` | `results/out42/` |
| 4.4 | NVT (Langevin, 300 K) stretching and comparison with NVE | `scripts/task44_cnt.py` | `results/out44/` |
| 4.5 | Bond-breaking energy and stress–strain curve at 300 K | `scripts/task45_cnt.py` | `results/task45_*/` |

Result folders are named after their run parameters, e.g. `dt0.1_deps0.01_trelax0.5_n12` =
time step 0.1 fs, strain increment 0.01, relaxation time 0.5 ps, 12 increments.
Each contains per-step energies (`energies_stepXX.csv`), geometries (`.xyz`), the minimisation
log and a `summary.json`. The five `task45_*` folders are separate Task 4.5 runs.
`logs/` holds the LAMMPS logs of the NVE and NVT runs.

`plot_helpers.ipynb` contains the plotting code used for the figures in the report.

## Requirements

- Linux (or WSL on Windows)
- LAMMPS built as a Python library with the `MANYBODY` package (for `pair_style airebo`)
- The `CH.airebo` potential file, with the environment variable `LAMMPS_POTENTIALS` pointing to its folder
- Python: `ase numpy matplotlib`

## Usage

The scripts write to `out42/`, `out44/` etc. relative to the working directory, so run them from `results/`:

```bash
cd results
python ../scripts/task42_cnt.py --help    # each script lists its options
python ../scripts/task42_cnt.py --dt 0.1 --deps 0.01 --trelax 0.5
python ../scripts/task43_plot.py
python ../scripts/task44_cnt.py
python ../scripts/task45_cnt.py
```
