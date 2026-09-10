"""Current-map analysis and parameter export."""

import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from . import core
from .preprocessing import prepare_map
from .coordinates import (
    basis_sign,
    convert_maps,
    convert_positions,
    direct_basis,
    reciprocal_basis,
    site_weights,
)
from .readout import compute_readout, summarize

PARAMETER_COLUMNS = [
    "name",
    "I_floor_A",
    "I_A",
    "I_B",
    "I_H",
    "c_AH",
    "c_B",
    "theta_deg",
    "r_A",
    "w",
    "eta",
    "snr",
    "resolved",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_inputs(
    input_dir,
    output_dir,
    *,
    sheet=0,
    line_axis=1,
    current_unit="A",
    scan_size=(2.5, 2.5),
    repair_suffixes=False,
):
    core.INPUT_ROOT_FOR_PROVENANCE = input_dir
    core.RAW_SHEET = sheet
    core.REPAIR_DUPLICATE_SUFFIXES = repair_suffixes
    core.NUMERIC_INPUT_MULTIPLIER = {"A": 1.0, "nA": 1e-9, "pA": 1e-12, "fA": 1e-15}[
        current_unit
    ]
    core.SCAN_SIZE_X_NM, core.SCAN_SIZE_Y_NM = scan_size
    files = core.list_input_files(input_dir)
    if len(files) < 2:
        raise ValueError("At least two input maps are required")
    hashes = [sha256(file) for file in files]
    pd.DataFrame(
        {
            "relative_path": [f.relative_to(input_dir).as_posix() for f in files],
            "sha256": hashes,
        }
    ).to_csv(output_dir / "input_manifest.csv", index=False)
    records, log = core.extract_all_images(files)
    log.to_csv(output_dir / "input_log.csv", index=False)
    if len(records) != len(files) or len({r.name for r in records}) != len(files):
        raise ValueError(
            "One or more inputs failed; see input_log.csv. No subset was analyzed."
        )
    if [sha256(file) for file in files] != hashes:
        raise ValueError("Input files changed while being read")
    folded_norm, folded_raw, floors, lattice = [], [], [], []
    for i, record in enumerate(records):
        corrected = prepare_map(record.data, line_axis)
        G, _, _ = core.find_lattice_vectors_fft(corrected)
        norm, _ = core.robust_zscore(corrected)
        cell, _ = core.fold_image_to_unit_cell(norm, G)
        cell, _ = core.robust_zscore(cell)
        folded_norm.append(cell)
        raw_cell, _ = core.fold_image_to_unit_cell(
            core.fill_nan_nearest(record.data), G
        )
        folded_raw.append(raw_cell)
        floors.append(
            {"name": record.name, "I_floor_A": core.current_floor(record.data)}
        )
        lattice.append(
            {
                "name": record.name,
                "g1_x": G[0, 0],
                "g1_y": G[0, 1],
                "g2_x": G[1, 0],
                "g2_y": G[1, 1],
            }
        )
        if (i + 1) % 25 == 0 or i + 1 == len(records):
            print(f"Folded {i+1}/{len(records)} maps", flush=True)
    aligned, shifts = core.align_unit_cell_maps(np.stack(folded_norm))
    raw = np.stack([core.roll2(cell, shift) for cell, shift in zip(folded_raw, shifts)])
    return (
        {"normalized": aligned, "raw_A": raw},
        pd.DataFrame(floors),
        pd.DataFrame(lattice),
        shifts,
    )


def analyze_sites(arrays, names, basis):
    maps = arrays["normalized"]
    if maps.shape != (len(names), 64, 64) or arrays["raw_A"].shape != maps.shape:
        raise ValueError("Expected matching 64x64 normalized and raw unit cells")
    choices = core.find_sites(maps, names)
    converted = {key: convert_maps(values, basis) for key, values in arrays.items()}
    rows, sites = [], []
    for i, name in enumerate(names):
        positions = core.refine_A_site(maps[i], choices[name])
        weights = {}
        for site in ("A", "B", "H"):
            position = convert_positions(positions[site], basis)
            weights[site] = site_weights((64, 64), position, basis)
            sites.append(
                {"name": name, "site": site, "u": position[0], "v": position[1]}
            )
        intensity = {
            site: float(np.sum(converted["normalized"][i] * weights[site]))
            for site in weights
        }
        currents = {
            site: float(np.sum(converted["raw_A"][i] * weights[site]))
            for site in weights
        }
        rows.append(
            {
                "name": name,
                **{f"I_{site}": value for site, value in intensity.items()},
                **core.contrast_coordinates(
                    intensity["A"], intensity["B"], intensity["H"]
                ),
                "r_A": core.contrast_amplitude(
                    currents["A"], currents["B"], currents["H"]
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(sites), converted


def finish(arrays, floors, lattice, shifts, output_dir, settings):
    basis = settings["basis"]
    names = floors.name.astype(str).tolist()
    motif, sites, converted = analyze_sites(arrays, names, basis)
    readout, modes = compute_readout(
        converted["normalized"],
        motif.theta_deg.to_numpy(),
        names,
        basis,
        settings["template_n"],
        settings["snr_threshold"],
    )
    parameters = motif.merge(floors, on="name", validate="one_to_one").merge(
        readout, on="name", validate="one_to_one"
    )
    parameters = parameters[PARAMETER_COLUMNS]
    parameters.to_csv(output_dir / "parameters.csv", index=False)
    sites.to_csv(output_dir / "sites.csv", index=False)
    modes.to_csv(output_dir / "fourier_modes.csv", index=False)
    summarize(parameters).to_csv(output_dir / "statistics.csv", index=False)
    np.savez_compressed(
        output_dir / "unit_cells.npz",
        names=np.array(names, dtype=str),
        basis=np.array(basis),
        direct_basis=direct_basis(basis),
        reciprocal_basis=reciprocal_basis(basis),
        **converted,
    )
    if len(lattice):
        lattice = lattice.copy()
        lattice[["g2_x", "g2_y"]] *= basis_sign(basis)
        lattice.to_csv(output_dir / "lattice.csv", index=False)
    if len(shifts):
        pd.DataFrame(
            {
                "name": names,
                "shift_v_pixels": [s[0] * basis_sign(basis) for s in shifts],
                "shift_u_pixels": [s[1] for s in shifts],
            }
        ).to_csv(output_dir / "alignment.csv", index=False)
    settings.update(
        {
            "n_maps": len(names),
            "n_selected": int(parameters.resolved.sum()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        }
    )
    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "n_maps": len(names),
                "n_selected": int(parameters.resolved.sum()),
                "median_snr": float(parameters.snr.median()),
            },
            indent=2,
        ),
        flush=True,
    )
    return parameters
