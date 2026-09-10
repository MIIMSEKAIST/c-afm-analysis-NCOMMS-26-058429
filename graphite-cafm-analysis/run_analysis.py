"""Analyze raw graphite current maps."""

import argparse
from pathlib import Path
import numpy as np
from cafm.coordinates import BASES
from cafm.pipeline import fold_inputs, finish


def sheet_value(text):
    return int(text) if text.isdigit() else text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sheet", type=sheet_value, default=0, help="Sheet name or zero-based index"
    )
    parser.add_argument(
        "--line-axis",
        type=int,
        choices=[0, 1],
        default=1,
        help="Fast-scan axis: 1 along rows; 0 along columns",
    )
    parser.add_argument("--basis", choices=BASES, default="direct120")
    parser.add_argument("--current-unit", choices=["A", "nA", "pA", "fA"], default="A")
    parser.add_argument("--scan-size-nm", nargs=2, type=float, default=[2.5, 2.5])
    parser.add_argument("--repair-duplicate-suffixes", action="store_true")
    parser.add_argument("--template-n", type=int, default=51)
    args = parser.parse_args(argv)
    src, out = (
        args.input_dir.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )
    if not src.is_dir():
        parser.error("Input directory does not exist")
    if out.is_relative_to(src):
        parser.error("Output directory must be outside the input directory")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error("Choose a new or empty output directory")
    if args.template_n < 1 or not all(
        np.isfinite(v) and v > 0 for v in args.scan_size_nm
    ):
        parser.error("Template count and scan dimensions must be positive")
    if (
        isinstance(args.sheet, str)
        and args.sheet.startswith("-")
        and args.sheet[1:].isdigit()
    ):
        parser.error("Sheet indices must be nonnegative")
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "basis": args.basis,
        "sheet": args.sheet,
        "line_axis": args.line_axis,
        "flatten_order": 1,
        "median_size": 3,
        "median_passes": 1,
        "median_edges": "reflect",
        "current_unit": args.current_unit,
        "scan_size_nm": args.scan_size_nm,
        "repair_duplicate_suffixes": args.repair_duplicate_suffixes,
        "template_n": args.template_n,
        "snr_threshold": 5.0,
    }
    data = fold_inputs(
        src,
        out,
        sheet=args.sheet,
        line_axis=args.line_axis,
        current_unit=args.current_unit,
        scan_size=args.scan_size_nm,
        repair_suffixes=args.repair_duplicate_suffixes,
    )
    finish(*data, out, settings)


if __name__ == "__main__":
    main()
