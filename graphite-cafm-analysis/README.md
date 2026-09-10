# Graphite C-AFM analysis

Lattice folding and A/B/H contrast analysis of raw current maps.

## Run

Python 3.12:

```bash
python -m pip install -r requirements.txt
python run_analysis.py --input-dir data/maps --output-dir results
```

Each XLSX/XLSM file supplies a header-free matrix from its first sheet; CSV
files supply numeric matrices. Currents default to amperes and the field of
view to 2.5 × 2.5 nm. Use `--sheet`, `--current-unit` and `--scan-size-nm` to
change these settings.

Each matrix row is flattened with a first-order line fit, followed by one
3×3 median pass with reflect edges. Use `--line-axis 0` for column-wise scan
lines. Supply raw maps, not maps that have already undergone this processing.
Files are processed in relative-filename order.

For first-row numeric strings carrying duplicate-column suffixes, add
`--repair-duplicate-suffixes`. Only suffixes with a matching unsuffixed value
in the same row are repaired; changes are recorded in `input_log.csv`.

## Output

`parameters.csv`: current floor, normalized A/B/H intensities, c_AH, c_B,
theta_deg, r_A, w, eta, snr and the selection flag. `I_floor_A` and `r_A`
are in amperes; `I_A`, `I_B` and `I_H` are normalized intensities.

`sites.csv` and `unit_cells.npz` store site coordinates and folded maps.
Lattice vectors, shifts, Fourier indices, statistics, settings and input
hashes are saved alongside them. Calculation definitions are in [METHODS.md](METHODS.md).

```bash
python -m unittest discover -s tests -v
```
