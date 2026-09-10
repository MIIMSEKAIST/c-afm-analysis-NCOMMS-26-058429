"""Current-map input, lattice folding and site extraction."""

from __future__ import annotations
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import ndimage

INPUT_ROOT_FOR_PROVENANCE = Path(".").resolve()
SEARCH_RECURSIVELY = True
INPUT_SUFFIXES = {".xlsx", ".xlsm", ".csv"}
OUTPUT_FOLDER_NAME = "analysis_results"
SCAN_SIZE_X_NM = 2.5
SCAN_SIZE_Y_NM = 2.5
GRAPHITE_LATTICE_A_NM = 0.246
USE_GRAPHITE_LATTICE_CONSTRAINT = True
LATTICE_RECIPROCAL_MAG_TOL = 0.45
UNIT_CELL_GRID = 64
SITE_AVERAGE_SIGMA_FRAC = 0.055
ORIGIN_GRID_SEARCH_N = 32
ALIGN_ITERATIONS = 3
SITE_BATCH_SIZE = 512
SITE_DTYPE = "float32"
A_REFINEMENT_RADIUS_FRAC = 0.045
A_REFINEMENT_GRID_STEPS = 11
A_REFINEMENT_SIGMA_SHARP_FRAC = 0.032
A_REFINEMENT_SIGMA_BROAD_FRAC = 0.09
A_REFINEMENT_PRIOR_WEIGHT = 0.02
A_REFINEMENT_ORDER_PENALTY_WEIGHT = 8.0
A_REFINEMENT_INTENSITY_WEIGHT = 0.15
A_REFINEMENT_MIN_SCORE_IMPROVEMENT = 0.0001
A_REFINEMENT_MAX_THETA_CHANGE_DEG = 25.0
A_REFINEMENT_ACCEPT_IF_ORDER_IMPROVES = True
SITE_GEOMETRY_MODEL = "graphite_single_hollow"
NUMERIC_INPUT_MULTIPLIER = 1.0
PARSE_TEXT_UNITS = True
MIN_IMAGE_HEIGHT = 16
MIN_IMAGE_WIDTH = 16
RAW_SHEET = 0
REPAIR_DUPLICATE_SUFFIXES = False
ROBUST_EPS = 1e-15
_SUPERSCRIPT_MAP = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁻": "-",
        "⁺": "+",
        "−": "-",
        "–": "-",
        "—": "-",
    }
)
CANDIDATE_BASES = {
    "graphite_hollow_13_23__23_13": {
        "atom_offsets": ((1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0 / 3.0)),
        "hollow_offset": (0.0, 0.0),
    },
    "graphite_hollow_13_13__23_23": {
        "atom_offsets": ((1.0 / 3.0, 1.0 / 3.0), (2.0 / 3.0, 2.0 / 3.0)),
        "hollow_offset": (0.0, 0.0),
    },
}


def list_input_files(input_path: Path) -> List[Path]:
    if not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_path}")
    candidates = input_path.rglob("*") if SEARCH_RECURSIVELY else input_path.glob("*")
    output_path = (input_path / OUTPUT_FOLDER_NAME).resolve()
    return sorted(
        (
            f
            for f in candidates
            if f.is_file()
            and f.suffix.lower() in INPUT_SUFFIXES
            and (not f.name.startswith("~$"))
            and (not f.resolve().is_relative_to(output_path))
        ),
        key=lambda f: f.relative_to(input_path).as_posix().casefold(),
    )


def _provenance_path(file_path: Path) -> str:
    resolved = Path(file_path).resolve()
    try:
        return resolved.relative_to(INPUT_ROOT_FOR_PROVENANCE).as_posix()
    except ValueError:
        return resolved.name


def parse_numeric_cell(x) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        value = float(x) * NUMERIC_INPUT_MULTIPLIER
        return value if math.isfinite(value) else np.nan
    if not isinstance(x, str) or not x.strip():
        return np.nan
    text = x.strip().replace("−", "-").replace("×", "x").replace("⋅", "x")
    text = re.sub(
        "10([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: "10^" + m.group(1).translate(_SUPERSCRIPT_MAP),
        text,
    )
    if "," in text:
        if not re.match(
            "^[+-]?\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?(?:\\s|[eEfpnumµμAa]|$)", text
        ):
            return np.nan
        text = text.replace(",", "")
    number = "[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?"
    power = f"(?:{number}\\s*[x*]\\s*)?10\\s*\\^\\s*[+-]?\\d+"
    match = re.fullmatch(
        f"(?P<value>{power}|{number})\\s*(?P<unit>[fpnumµμ]?[Aa])?", text
    )
    if match is None:
        return np.nan
    unit = match.group("unit")
    if unit is not None and (not PARSE_TEXT_UNITS):
        return np.nan
    factors = {
        "a": 1.0,
        "fa": 1e-15,
        "pa": 1e-12,
        "na": 1e-09,
        "ua": 1e-06,
        "µa": 1e-06,
        "μa": 1e-06,
        "ma": 0.001,
    }
    multiplier = factors[unit.lower()] if unit else NUMERIC_INPUT_MULTIPLIER
    value_text = match.group("value")
    try:
        if "^" in value_text:
            power_match = re.fullmatch(
                f"(?:(?P<coefficient>{number})\\s*[x*]\\s*)?10\\s*\\^\\s*(?P<exponent>[+-]?\\d+)",
                value_text,
            )
            coefficient = float(power_match.group("coefficient") or "1")
            value = coefficient * 10.0 ** int(power_match.group("exponent"))
        else:
            value = float(value_text)
        value *= multiplier
        return value if math.isfinite(value) else np.nan
    except (ValueError, OverflowError):
        return np.nan


def robust_zscore(
    a: np.ndarray, eps: float = ROBUST_EPS
) -> Tuple[np.ndarray, Dict[str, float]]:
    arr = np.asarray(a, dtype=float)
    med = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        scale = float(np.nanstd(arr))
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    z = (arr - med) / scale
    return (z, {"median": med, "mad": mad, "scale": scale})


def fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
    a = np.array(arr, dtype=float, copy=True)
    if not np.isnan(a).any():
        return a
    if np.all(np.isnan(a)):
        return np.zeros_like(a)
    mask = np.isnan(a)
    idx = ndimage.distance_transform_edt(
        mask, return_distances=False, return_indices=True
    )
    a[mask] = a[tuple(idx[:, mask])]
    return a


def remove_linear_plane(img: np.ndarray) -> np.ndarray:
    z = np.asarray(img, dtype=float)
    ny, nx = z.shape
    yy, xx = np.indices(z.shape)
    valid = np.isfinite(z)
    if valid.sum() < 10:
        return np.nan_to_num(z, nan=0.0)
    A = np.column_stack([xx[valid].ravel(), yy[valid].ravel(), np.ones(valid.sum())])
    b = z[valid].ravel()
    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        plane = coef[0] * xx + coef[1] * yy + coef[2]
        return z - plane
    except Exception:
        return z - np.nanmedian(z)


@dataclass
class ImageRecord:
    name: str
    file: str
    sheet: str
    data: np.ndarray


def repair_duplicate_suffixes(
    frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    numeric = "[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?"
    peers = {
        float(str(v).strip())
        for v in frame.iloc[0]
        if re.fullmatch(numeric, str(v).strip())
    }
    repaired = frame.copy()
    log = []
    for column, value in enumerate(frame.iloc[0]):
        if re.fullmatch(numeric, str(value).strip()):
            continue
        match = re.fullmatch("(" + numeric + ")\\.[1-9]\\d*", str(value).strip())
        if match and float(match[1]) in peers:
            repaired.iat[0, column] = float(match[1])
            log.append(
                {
                    "row_1based": 1,
                    "column_1based": column + 1,
                    "original_value": str(value),
                    "repaired_value": float(match[1]),
                }
            )
    return (repaired, log)


def _sheet_block_from_dataframe(df: pd.DataFrame) -> np.ndarray:
    if all(
        (
            pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype)
            for dtype in df.dtypes
        )
    ):
        num = df.to_numpy(dtype=float) * NUMERIC_INPUT_MULTIPLIER
    else:
        num = df.map(parse_numeric_cell).to_numpy(dtype=float)
    nonempty = df.notna().to_numpy()
    if np.any(nonempty & ~np.isfinite(num)):
        raise ValueError(
            "Excel input must be a current matrix without headers or coordinate columns"
        )
    return num


def extract_raw_map(
    file_path: Path,
) -> Tuple[List[ImageRecord], List[Dict[str, object]]]:
    source = _provenance_path(file_path)
    sheet = ""
    repair_log = []
    try:
        if file_path.suffix.lower() == ".csv":
            block = (
                np.loadtxt(file_path, delimiter=",", ndmin=2) * NUMERIC_INPUT_MULTIPLIER
            )
        else:
            with pd.ExcelFile(file_path, engine="openpyxl") as workbook:
                sheet = (
                    workbook.sheet_names[RAW_SHEET]
                    if isinstance(RAW_SHEET, int)
                    else RAW_SHEET
                )
                frame = workbook.parse(sheet_name=sheet, header=None)
                if REPAIR_DUPLICATE_SUFFIXES:
                    frame, repairs = repair_duplicate_suffixes(frame)
                    repair_log.extend(
                        (
                            {
                                "file": source,
                                "sheet": sheet,
                                "status": "duplicate_suffix_repaired",
                                **row,
                            }
                            for row in repairs
                        )
                    )
                block = _sheet_block_from_dataframe(frame)
        if (
            block is None
            or block.ndim != 2
            or block.shape[0] < MIN_IMAGE_HEIGHT
            or (block.shape[1] < MIN_IMAGE_WIDTH)
        ):
            raise ValueError(
                "The selected input must contain a current matrix of at least 16 x 16 pixels"
            )
        if np.isinf(block).any() or np.count_nonzero(np.isfinite(block)) < 10:
            raise ValueError(
                "The current map contains infinite values or too few finite pixels"
            )
        record = ImageRecord(name=source, file=source, sheet=sheet, data=block)
        return (
            [record],
            repair_log
            + [
                {
                    "file": source,
                    "sheet": sheet,
                    "status": "raw_map_ok",
                    "input_mode": "single",
                    "height": block.shape[0],
                    "width": block.shape[1],
                    "finite_fraction": float(np.isfinite(block).mean()),
                }
            ],
        )
    except Exception as exc:
        return (
            [],
            [
                {
                    "file": source,
                    "sheet": sheet,
                    "status": "read_error",
                    "message": str(exc),
                }
            ],
        )


def extract_all_images(files: Sequence[Path]) -> Tuple[List[ImageRecord], pd.DataFrame]:
    all_records: List[ImageRecord] = []
    logs: List[Dict[str, object]] = []
    for index, f in enumerate(files, 1):
        recs, lgs = extract_raw_map(f)
        all_records.extend(recs)
        logs.extend(lgs)
        if index % 50 == 0 or index == len(files):
            print(f"Read {index}/{len(files)} current maps", flush=True)
    return (all_records, pd.DataFrame(logs))


def fft_preprocess(img: np.ndarray) -> np.ndarray:
    z, _ = robust_zscore(img)
    z = remove_linear_plane(z)
    sigma = max(2.0, min(z.shape) / 16.0)
    z_hp = z - ndimage.gaussian_filter(z, sigma=sigma, mode="nearest")
    z_hp, _ = robust_zscore(z_hp)
    wy = np.hanning(z_hp.shape[0])
    wx = np.hanning(z_hp.shape[1])
    return np.nan_to_num(z_hp, nan=0.0) * wy[:, None] * wx[None, :]


def expected_reciprocal_cycles_per_nm() -> float:
    return 2.0 / (math.sqrt(3.0) * GRAPHITE_LATTICE_A_NM)


def find_lattice_vectors_fft(
    img: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    ny, nx = img.shape
    prep = fft_preprocess(img)
    F = np.fft.fftshift(np.fft.fft2(prep))
    power = np.abs(F) ** 2
    cy, cx = (ny // 2, nx // 2)
    power[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0.0
    yidx = np.arange(ny) - cy
    xidx = np.arange(nx) - cx
    KY_idx, KX_idx = np.meshgrid(yidx, xidx, indexing="ij")
    fx_nm = KX_idx / SCAN_SIZE_X_NM
    fy_nm = KY_idx / SCAN_SIZE_Y_NM
    mag_nm = np.sqrt(fx_nm**2 + fy_nm**2)
    exp_mag = expected_reciprocal_cycles_per_nm()
    if USE_GRAPHITE_LATTICE_CONSTRAINT:
        lo = exp_mag * (1.0 - LATTICE_RECIPROCAL_MAG_TOL)
        hi = exp_mag * (1.0 + LATTICE_RECIPROCAL_MAG_TOL)
        band = (mag_nm >= lo) & (mag_nm <= hi)
    else:
        band = mag_nm > 0.08 * max(nx / SCAN_SIZE_X_NM, ny / SCAN_SIZE_Y_NM)
    local_max = power == ndimage.maximum_filter(power, size=5, mode="nearest")
    mask = band & local_max & np.isfinite(power)
    ys, xs = np.where(mask)
    if ys.size < 6:
        band = (mag_nm >= exp_mag * 0.35) & (mag_nm <= exp_mag * 1.85)
        mask = band & local_max & np.isfinite(power)
        ys, xs = np.where(mask)
    if ys.size < 2:
        raise RuntimeError(
            "FFT lattice peak detection failed: not enough peaks in graphite band."
        )
    vals = power[ys, xs]
    order = np.argsort(vals)[::-1]
    max_candidates = min(80, order.size)
    candidates = []
    for idx in order[:max_candidates]:
        y, x = (int(ys[idx]), int(xs[idx]))
        kx_i = float(KX_idx[y, x])
        ky_i = float(KY_idx[y, x])
        if abs(kx_i) < 1 and abs(ky_i) < 1:
            continue
        candidates.append(
            {
                "kx_idx": kx_i,
                "ky_idx": ky_i,
                "power": float(power[y, x]),
                "mag_nm": float(mag_nm[y, x]),
                "y": y,
                "x": x,
            }
        )
    if len(candidates) < 2:
        raise RuntimeError(
            "FFT lattice peak detection failed after candidate filtering."
        )
    best = None
    best_score = -np.inf
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            c1, c2 = (candidates[i], candidates[j])
            v1_nm = np.array(
                [c1["kx_idx"] / SCAN_SIZE_X_NM, c1["ky_idx"] / SCAN_SIZE_Y_NM],
                dtype=float,
            )
            v2_nm = np.array(
                [c2["kx_idx"] / SCAN_SIZE_X_NM, c2["ky_idx"] / SCAN_SIZE_Y_NM],
                dtype=float,
            )
            n1, n2 = (np.linalg.norm(v1_nm), np.linalg.norm(v2_nm))
            if n1 == 0 or n2 == 0:
                continue
            cosang = float(np.clip(np.dot(v1_nm, v2_nm) / (n1 * n2), -1.0, 1.0))
            angle = math.degrees(math.acos(cosang))
            if angle < 35 or angle > 145:
                continue
            angle_penalty = min(abs(angle - 60.0), abs(angle - 120.0)) / 30.0
            mag_penalty = (abs(n1 - exp_mag) + abs(n2 - exp_mag)) / max(exp_mag, 1e-09)
            ratio_penalty = abs(math.log((n1 + 1e-09) / (n2 + 1e-09)))
            pscore = math.log(c1["power"] + 1.0) + math.log(c2["power"] + 1.0)
            score = (
                pscore - 3.0 * angle_penalty - 3.0 * mag_penalty - 2.0 * ratio_penalty
            )
            if score > best_score:
                best_score = score
                best = (c1, c2, angle, n1, n2)
    if best is None:
        raise RuntimeError(
            "FFT lattice peak detection failed: no suitable non-collinear peak pair."
        )
    c1, c2, angle, n1, n2 = best
    G_raw_selected = np.array(
        [
            [2.0 * np.pi * c1["kx_idx"] / nx, 2.0 * np.pi * c1["ky_idx"] / ny],
            [2.0 * np.pi * c2["kx_idx"] / nx, 2.0 * np.pi * c2["ky_idx"] / ny],
        ],
        dtype=float,
    )
    G, canonical_diag = canonicalize_reciprocal_basis(G_raw_selected)
    dx = SCAN_SIZE_X_NM / nx
    dy = SCAN_SIZE_Y_NM / ny
    G_nm = G.copy()
    G_nm[:, 0] /= dx
    G_nm[:, 1] /= dy
    try:
        A_nm = 2.0 * np.pi * np.linalg.inv(G_nm)
        a1 = float(np.linalg.norm(A_nm[:, 0]))
        a2 = float(np.linalg.norm(A_nm[:, 1]))
        aang = float(
            math.degrees(
                math.acos(
                    np.clip(np.dot(A_nm[:, 0], A_nm[:, 1]) / (a1 * a2), -1.0, 1.0)
                )
            )
        )
        area = float(abs(np.linalg.det(A_nm)))
    except Exception:
        a1 = a2 = aang = area = np.nan
    diag = {
        "fft_peak1_kx_idx": c1["kx_idx"],
        "fft_peak1_ky_idx": c1["ky_idx"],
        "fft_peak2_kx_idx": c2["kx_idx"],
        "fft_peak2_ky_idx": c2["ky_idx"],
        "fft_pair_angle_deg_reciprocal": float(angle),
        "fft_peak1_mag_cycles_per_nm": float(n1),
        "fft_peak2_mag_cycles_per_nm": float(n2),
        "expected_mag_cycles_per_nm": float(exp_mag),
        "lattice_a1_nm": a1,
        "lattice_a2_nm": a2,
        "lattice_angle_deg": aang,
        "lattice_area_nm2": area,
        "fft_pair_score": float(best_score),
        **canonical_diag,
    }
    return (G, diag, power)


def canonicalize_reciprocal_basis(G: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    G = np.asarray(G, dtype=float)
    cand = []
    for swap in (False, True):
        B = G[[1, 0], :].copy() if swap else G.copy()
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                C = B.copy()
                C[0] *= s1
                C[1] *= s2
                n1 = np.linalg.norm(C[0])
                n2 = np.linalg.norm(C[1])
                if n1 <= 0 or n2 <= 0:
                    continue
                angle = math.degrees(
                    math.acos(np.clip(np.dot(C[0], C[1]) / (n1 * n2), -1.0, 1.0))
                )
                det = float(np.linalg.det(C))
                a1 = math.degrees(math.atan2(C[0, 1], C[0, 0]))
                orient_pen = abs((a1 + 180.0) % 360.0 - 180.0) / 180.0
                score = (
                    -abs(angle - 60.0) - (0.0 if det > 0 else 100.0) - 0.01 * orient_pen
                )
                cand.append((score, C, angle, det, swap, s1, s2, a1))
    if not cand:
        return (G, {"fft_basis_canonicalized": False})
    cand.sort(key=lambda x: x[0], reverse=True)
    _, C, angle, det, swap, s1, s2, a1 = cand[0]
    return (
        C,
        {
            "fft_basis_canonicalized": True,
            "fft_canonical_angle_deg": float(angle),
            "fft_canonical_det": float(det),
            "fft_canonical_swapped": bool(swap),
            "fft_canonical_sign1": float(s1),
            "fft_canonical_sign2": float(s2),
            "fft_canonical_g1_angle_deg": float(a1),
        },
    )


def fold_image_to_unit_cell(
    img: np.ndarray,
    G: np.ndarray,
    grid: int = UNIT_CELL_GRID,
    phase_offset: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(img, dtype=float)
    ny, nx = arr.shape
    yy, xx = np.indices(arr.shape)
    u = (G[0, 0] * xx + G[0, 1] * yy) / (2.0 * np.pi) + float(phase_offset[0])
    v = (G[1, 0] * xx + G[1, 1] * yy) / (2.0 * np.pi) + float(phase_offset[1])
    u = np.mod(u, 1.0)
    v = np.mod(v, 1.0)
    iu = np.floor(u * grid).astype(int) % grid
    iv = np.floor(v * grid).astype(int) % grid
    accum = np.zeros((grid, grid), dtype=float)
    count = np.zeros((grid, grid), dtype=float)
    vals = np.nan_to_num(arr, nan=np.nanmedian(arr))
    np.add.at(accum, (iv.ravel(), iu.ravel()), vals.ravel())
    np.add.at(count, (iv.ravel(), iu.ravel()), 1.0)
    out = np.divide(accum, count, out=np.full_like(accum, np.nan), where=count > 0)
    out = fill_nan_nearest(out)
    return (out, count)


def roll2(a: np.ndarray, shift: Tuple[int, int]) -> np.ndarray:
    return np.roll(np.roll(a, int(shift[0]), axis=0), int(shift[1]), axis=1)


def corrcoef2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a).ravel()
    bb = np.asarray(b).ravel()
    aa = aa - np.nanmean(aa)
    bb = bb - np.nanmean(bb)
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    if den <= 0:
        return 0.0
    return float(np.dot(aa, bb) / den)


def best_integer_shift_to_ref(img: np.ndarray, ref: np.ndarray) -> Tuple[int, int]:
    f_ref = np.fft.fft2(ref - np.mean(ref))
    f_img = np.fft.fft2(img - np.mean(img))
    corr = np.fft.ifft2(f_ref * np.conj(f_img)).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    sy, sx = (int(peak[0]), int(peak[1]))
    if sy > img.shape[0] // 2:
        sy -= img.shape[0]
    if sx > img.shape[1] // 2:
        sx -= img.shape[1]
    cand1 = (sy, sx)
    cand2 = (-sy, -sx)
    if corrcoef2(roll2(img, cand2), ref) > corrcoef2(roll2(img, cand1), ref):
        return cand2
    return cand1


def align_unit_cell_maps(
    maps: np.ndarray, iterations: int = ALIGN_ITERATIONS
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    arr = np.array(maps, dtype=float, copy=True)
    n = arr.shape[0]
    ref = arr[0] - np.mean(arr[0])
    cumulative = [(0, 0) for _ in range(n)]
    aligned = arr.copy()
    for _ in range(max(1, iterations)):
        new_aligned = []
        for i in range(n):
            sh = best_integer_shift_to_ref(aligned[i], ref)
            new_map = roll2(aligned[i], sh)
            cumulative[i] = (cumulative[i][0] + sh[0], cumulative[i][1] + sh[1])
            new_aligned.append(new_map)
        aligned = np.stack(new_aligned, axis=0)
        ref = np.mean(aligned, axis=0)
        ref = ref - np.mean(ref)
    return (aligned, cumulative)


def site_search_bases():
    return tuple(CANDIDATE_BASES)


def mod1_pair(p: Tuple[float, float]) -> Tuple[float, float]:
    return (float(p[0] % 1.0), float(p[1] % 1.0))


def positions_from_origin_basis(
    origin: Tuple[float, float], basis_name: str
) -> Dict[str, Tuple[float, float]]:
    basis = CANDIDATE_BASES[basis_name]
    atom_offsets = basis["atom_offsets"]
    h_off = basis["hollow_offset"]
    ou, ov = origin
    geoA = mod1_pair((ou + atom_offsets[0][0], ov + atom_offsets[0][1]))
    geoB = mod1_pair((ou + atom_offsets[1][0], ov + atom_offsets[1][1]))
    geoH = mod1_pair((ou + h_off[0], ov + h_off[1]))
    return {"geoA": geoA, "geoB": geoB, "H": geoH}


def periodic_gaussian_weight(
    grid: int, pos: Tuple[float, float], sigma_frac: float = SITE_AVERAGE_SIGMA_FRAC
) -> np.ndarray:
    u0, v0 = pos
    coords = (np.arange(grid) + 0.5) / grid
    U, V = np.meshgrid(coords, coords, indexing="xy")
    du0 = (U - u0 + 0.5) % 1.0 - 0.5
    dv0 = (V - v0 + 0.5) % 1.0 - 0.5
    best = np.full_like(du0, np.inf, dtype=float)
    for su in (-1.0, 0.0, 1.0):
        for sv in (-1.0, 0.0, 1.0):
            du = du0 + su
            dv = dv0 + sv
            d2 = du**2 + dv**2 + du * dv
            best = np.minimum(best, d2)
    w = np.exp(-best / (2.0 * sigma_frac**2))
    s = float(w.sum())
    if s <= 0:
        return np.ones((grid, grid)) / (grid * grid)
    return w / s


def site_mean(
    unit_map: np.ndarray,
    pos: Tuple[float, float],
    sigma_frac: float = SITE_AVERAGE_SIGMA_FRAC,
) -> float:
    w = periodic_gaussian_weight(unit_map.shape[0], pos, sigma_frac)
    return float(np.sum(w * unit_map))


def site_pick_score_from_deltas(d_ba, d_ah, ordered=None, stability=0.0):
    visibility = d_ba + d_ah
    negative = 5.0 * (np.maximum(0.0, -d_ba) + np.maximum(0.0, -d_ah))
    score = visibility - negative
    if ordered is not None:
        score = score + 0.1 * np.asarray(ordered, dtype=float)
    if stability is not None:
        score = score + stability
    return score


def build_site_candidate_table() -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rows: List[Dict[str, object]] = []
    arrs: Dict[str, List[float]] = {
        "geoA_u": [],
        "geoA_v": [],
        "geoB_u": [],
        "geoB_v": [],
        "H_u": [],
        "H_v": [],
    }
    grid = np.arange(ORIGIN_GRID_SEARCH_N, dtype=float) / ORIGIN_GRID_SEARCH_N
    for basis in site_search_bases():
        for ou in grid:
            for ov in grid:
                pos = positions_from_origin_basis((float(ou), float(ov)), basis)
                idx = len(rows)
                rows.append(
                    {
                        "candidate_index": idx,
                        "basis": basis,
                        "site_geometry_model": SITE_GEOMETRY_MODEL,
                        "origin_definition": "single_geometric_hollow",
                        "origin_u": float(ou),
                        "origin_v": float(ov),
                        "geoA_u": pos["geoA"][0],
                        "geoA_v": pos["geoA"][1],
                        "geoB_u": pos["geoB"][0],
                        "geoB_v": pos["geoB"][1],
                        "geoH_u": pos["H"][0],
                        "geoH_v": pos["H"][1],
                    }
                )
                arrs["geoA_u"].append(pos["geoA"][0])
                arrs["geoA_v"].append(pos["geoA"][1])
                arrs["geoB_u"].append(pos["geoB"][0])
                arrs["geoB_v"].append(pos["geoB"][1])
                arrs["H_u"].append(pos["H"][0])
                arrs["H_v"].append(pos["H"][1])
    return (
        pd.DataFrame(rows),
        {k: np.asarray(v, dtype=np.float32) for k, v in arrs.items()},
    )


def _site_values_vectorized_for_positions(
    maps_ref: np.ndarray,
    pos_u: np.ndarray,
    pos_v: np.ndarray,
    xp,
    batch_size: int = SITE_BATCH_SIZE,
) -> np.ndarray:
    maps_np = np.asarray(maps_ref, dtype=np.float32)
    if maps_np.ndim == 2:
        maps_np = maps_np[None, :, :]
    n_maps, grid_h, grid_w = maps_np.shape
    if grid_h != grid_w:
        raise ValueError(
            "Folded unit-cell maps must be square for site-search acceleration."
        )
    grid = int(grid_h)
    n_cand = int(len(pos_u))
    dtype = np.float32 if SITE_DTYPE == "float32" else np.float64
    coords = ((np.arange(grid, dtype=dtype) + dtype(0.5)) / dtype(grid)).astype(dtype)
    U, V = np.meshgrid(coords, coords, indexing="xy")
    pix_u_np = U.reshape(-1)
    pix_v_np = V.reshape(-1)
    maps_flat_np = maps_np.reshape(n_maps, -1)
    if not np.isfinite(maps_flat_np).all():
        med = np.nanmedian(maps_flat_np, axis=1)
        inds = np.where(~np.isfinite(maps_flat_np))
        maps_flat_np = maps_flat_np.copy()
        maps_flat_np[inds] = np.take(med, inds[0])
    pix_u = xp.asarray(pix_u_np, dtype=dtype)
    pix_v = xp.asarray(pix_v_np, dtype=dtype)
    maps_flat_T = xp.asarray(maps_flat_np.T, dtype=dtype)
    out_chunks: List[np.ndarray] = []
    sigma2 = dtype(2.0 * SITE_AVERAGE_SIGMA_FRAC**2)
    for start in range(0, n_cand, int(batch_size)):
        stop = min(n_cand, start + int(batch_size))
        pu = xp.asarray(pos_u[start:stop], dtype=dtype)[:, None]
        pv = xp.asarray(pos_v[start:stop], dtype=dtype)[:, None]
        du0 = (pix_u[None, :] - pu + dtype(0.5)) % dtype(1.0) - dtype(0.5)
        dv0 = (pix_v[None, :] - pv + dtype(0.5)) % dtype(1.0) - dtype(0.5)
        best = xp.full(du0.shape, xp.inf, dtype=dtype)
        for su in (-1.0, 0.0, 1.0):
            for sv in (-1.0, 0.0, 1.0):
                du = du0 + dtype(su)
                dv = dv0 + dtype(sv)
                d2 = du * du + dv * dv + du * dv
                best = xp.minimum(best, d2)
        W = xp.exp(-best / sigma2)
        W = W / (xp.sum(W, axis=1, keepdims=True) + dtype(1e-30))
        vals = W @ maps_flat_T
        out_chunks.append(vals.astype(np.float64, copy=False))
        del pu, pv, du0, dv0, best, W, vals
    return np.vstack(out_chunks)


def _site_distance_squared(du, dv):
    return du * du + dv * dv + du * dv


def refine_A_site(
    map_arr: np.ndarray, positions_role: Dict[str, Tuple[float, float]]
) -> Dict[str, Tuple[float, float]]:
    pos0 = {k: tuple(v) for k, v in positions_role.items()}
    A0 = tuple(pos0["A"])
    B = tuple(pos0["B"])
    H = tuple(pos0["H"])
    radius = float(A_REFINEMENT_RADIUS_FRAC)
    nsteps = int(A_REFINEMENT_GRID_STEPS)
    if nsteps < 3 or radius <= 0:
        return pos0
    IB = site_mean(map_arr, B, SITE_AVERAGE_SIGMA_FRAC)
    IH = site_mean(map_arr, H, SITE_AVERAGE_SIGMA_FRAC)
    IA0 = site_mean(map_arr, A0, SITE_AVERAGE_SIGMA_FRAC)
    coords0 = contrast_coordinates(IA0, IB, IH)
    theta0 = coords0.get("theta_deg", np.nan)
    offsets = np.linspace(-radius, radius, nsteps)
    cand_rows: List[Dict[str, float]] = []
    best_row: Optional[Dict[str, float]] = None
    best_score = -np.inf
    origin_score = np.nan
    for du in offsets:
        for dv in offsets:
            d_hex = math.sqrt(max(0.0, _site_distance_squared(du, dv)))
            if d_hex > radius * 1.0001:
                continue
            A = ((A0[0] + float(du)) % 1.0, (A0[1] + float(dv)) % 1.0)
            IA = site_mean(map_arr, A, SITE_AVERAGE_SIGMA_FRAC)
            IA_sharp = site_mean(map_arr, A, A_REFINEMENT_SIGMA_SHARP_FRAC)
            IA_broad = site_mean(map_arr, A, A_REFINEMENT_SIGMA_BROAD_FRAC)
            peakness = IA_sharp - IA_broad
            d_ba = IB - IA
            d_ah = IA - IH
            order_pen = max(0.0, -d_ba) ** 2 + max(0.0, -d_ah) ** 2
            prior_pen = (d_hex / max(radius, 1e-12)) ** 2
            score = (
                peakness
                + float(A_REFINEMENT_INTENSITY_WEIGHT) * IA
                - float(A_REFINEMENT_ORDER_PENALTY_WEIGHT) * order_pen
                - float(A_REFINEMENT_PRIOR_WEIGHT) * prior_pen
            )
            coords = contrast_coordinates(IA, IB, IH)
            row = {
                "A_u": A[0],
                "A_v": A[1],
                "shift_frac": float(d_hex),
                "theta_deg": float(coords["theta_deg"]),
                "score": float(score),
                "ordered": float(d_ba > 0 and d_ah > 0),
            }
            cand_rows.append(row)
            if abs(du) < 1e-12 and abs(dv) < 1e-12:
                origin_score = float(score)
            if np.isfinite(score) and score > best_score:
                best_score = float(score)
                best_row = row
    if best_row is None:
        return pos0
    if not np.isfinite(origin_score):
        origin_score = float(min(cand_rows, key=lambda r: r["shift_frac"])["score"])
    theta_best = float(best_row.get("theta_deg", np.nan))
    theta_change = (
        theta_best - theta0
        if np.isfinite(theta_best) and np.isfinite(theta0)
        else np.nan
    )
    orig_ordered = bool(IB - IA0 > 0 and IA0 - IH > 0)
    best_ordered = bool(best_row.get("ordered", 0.0) > 0.5)
    score_improvement = float(best_score - origin_score)
    accept = bool(score_improvement >= float(A_REFINEMENT_MIN_SCORE_IMPROVEMENT))
    if A_REFINEMENT_ACCEPT_IF_ORDER_IMPROVES and (not orig_ordered) and best_ordered:
        accept = True
    if np.isfinite(theta_change) and abs(theta_change) > float(
        A_REFINEMENT_MAX_THETA_CHANGE_DEG
    ):
        accept = False
    pos_ref = dict(pos0)
    if accept:
        pos_ref["A"] = (float(best_row["A_u"]), float(best_row["A_v"]))
    return pos_ref


def current_floor(raw):
    values = np.asarray(raw, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No finite current values")
    return float(np.nanpercentile(values, 5.0))


def contrast_coordinates(IA, IB, IH):
    c_AH = (IA + IB - 2.0 * IH) / math.sqrt(6.0)
    c_B = (IB - IA) / math.sqrt(2.0)
    theta = math.degrees(math.atan2(abs(c_B), c_AH))
    return {"c_AH": c_AH, "c_B": c_B, "theta_deg": theta}


def contrast_amplitude(IA, IB, IH):
    currents = np.array([IA, IB, IH], dtype=float)
    return float(np.sqrt(np.sum((currents - currents.mean()) ** 2)))


def find_sites(maps, names):
    maps = np.asarray(maps, dtype=float)
    if maps.ndim != 3 or len(maps) != len(names) or (not np.isfinite(maps).all()):
        raise ValueError("Expected one finite unit cell per acquisition")
    candidates, positions = build_site_candidate_table()
    a = _site_values_vectorized_for_positions(
        maps, positions["geoA_u"], positions["geoA_v"], np
    )
    b = _site_values_vectorized_for_positions(
        maps, positions["geoB_u"], positions["geoB_v"], np
    )
    h = _site_values_vectorized_for_positions(
        maps, positions["H_u"], positions["H_v"], np
    )
    bright_a = a >= b
    IB = np.where(bright_a, a, b)
    IA = np.where(bright_a, b, a)
    score = site_pick_score_from_deltas(
        IB - IA, IA - h, ((IB > IA) & (IA > h)).astype(float)
    )
    indices = np.nanargmax(score, axis=0)
    result = {}
    for j, name in enumerate(names):
        index = int(indices[j])
        row = candidates.iloc[index]
        first = (float(row.geoA_u), float(row.geoA_v))
        second = (float(row.geoB_u), float(row.geoB_v))
        result[name] = {
            "A": second if bright_a[index, j] else first,
            "B": first if bright_a[index, j] else second,
            "H": (float(row.geoH_u), float(row.geoH_v)),
        }
    return result
