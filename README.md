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
lines.
Files are processed in relative-filename order.

```bash
python -m unittest discover -s tests -v
```
