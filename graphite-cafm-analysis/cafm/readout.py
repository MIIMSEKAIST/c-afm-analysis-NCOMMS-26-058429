"""Fourier attenuation descriptors and amplitude selection."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .coordinates import (
    convert_indices,
    fourier_coefficients,
    q_squared,
    reciprocal_cartesian,
)

EPS = 1e-300
SIGNAL_HK = [(-2, -1), (-1, -2), (-1, 1), (1, -1), (1, 2), (2, 1)]
NOISE_HK = [
    (h, k) for h in range(-5, 6) for k in range(-5, 6) if h * h - h * k + k * k == 7
]


def shell_modes(basis):
    groups = []
    for q2 in (1, 3, 4):
        indices = [
            (h, k)
            for h in range(-4, 5)
            for k in range(-4, 5)
            if h * h + h * k + k * k == q2
        ]
        indices.sort(key=lambda hk: np.arctan2((2 * hk[1] + hk[0]) / np.sqrt(3), hk[0]))
        groups.append(convert_indices(indices, basis))
    return np.concatenate(groups)


def weighted_median(x, weights):
    """Weighted median interpolated at cumulative weight 0.5."""
    x, weights = np.asarray(x, float), np.asarray(weights, float)
    if (
        x.shape != weights.shape
        or not np.isfinite(x).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError("Expected finite values and positive matching weights")
    order = np.argsort(x)
    return float(np.interp(0.5, np.cumsum(weights[order]) / weights.sum(), x[order]))


def neighbors(theta, i, count=51):
    if len(theta) < 2 or not np.isfinite(theta).all() or count < 1:
        raise ValueError(
            "At least two finite theta values and a positive neighbor count are required"
        )
    delta = np.abs(theta - theta[i])
    delta[i] = np.inf
    indices = np.argsort(delta)[: min(count, len(theta) - 1)]
    scale = max(float(np.median(delta[indices])), 1.0)
    weights = np.exp(-0.5 * (delta[indices] / scale) ** 2)
    return indices, weights / weights.sum()


def amplitude_snr(maps, basis):
    signal = np.abs(fourier_coefficients(maps, convert_indices(SIGNAL_HK, basis))).mean(
        axis=1
    )
    noise = np.median(
        np.abs(fourier_coefficients(maps, convert_indices(NOISE_HK, basis))), axis=1
    )
    score = np.divide(signal, noise, out=np.full_like(signal, np.nan), where=noise > 0)
    return signal, noise, score


def compute_readout(maps, theta, names, basis, template_n=51, threshold=5.0):
    theta = np.asarray(theta, dtype=float)
    if (
        len(maps) != len(theta)
        or len(names) != len(theta)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Maps, theta and unique names must correspond one-to-one")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("SNR threshold must be finite and positive")
    hk = shell_modes(basis)
    q2 = q_squared(hk, basis)
    g = reciprocal_cartesian(hk, basis)
    logamp = np.log(np.maximum(np.abs(fourier_coefficients(maps, hk)), EPS))
    logratio = np.median(logamp[:, q2 == 3], axis=1) - np.median(
        logamp[:, q2 == 1], axis=1
    )
    design = np.column_stack(
        [np.ones(len(hk)), q2 - 1, g[:, 0] ** 2 - g[:, 1] ** 2, 2 * g[:, 0] * g[:, 1]]
    )
    signal, noise, snr = amplitude_snr(maps, basis)
    rows = []
    for i, name in enumerate(names):
        idx, weights = neighbors(theta, i, template_n)
        ratio_reference = weighted_median(logratio[idx], weights)
        reference = np.array(
            [weighted_median(logamp[idx, j], weights) for j in range(len(hk))]
        )
        y = -(logamp[i] - reference)
        ref_amp = np.exp(reference)
        mode_weights = np.ones(len(hk)) / 6
        mode_weights *= np.clip(
            ref_amp / np.median(ref_amp[np.isfinite(ref_amp) & (ref_amp > 0)]), 0.2, 5.0
        )
        sw = np.sqrt(mode_weights / mode_weights.mean())
        coef, _, rank, _ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
        if rank != 4:
            raise ValueError(f"Rank-deficient readout fit for {name}")
        rows.append(
            {
                "name": name,
                "w": -0.5 * (logratio[i] - ratio_reference),
                "eta": float(np.hypot(coef[2], coef[3])),
                "snr": snr[i],
                "resolved": bool(np.isfinite(snr[i]) and snr[i] >= threshold),
            }
        )
    modes = []
    for purpose, indices in [
        ("readout", hk),
        ("snr_signal", convert_indices(SIGNAL_HK, basis)),
        ("snr_noise", convert_indices(NOISE_HK, basis)),
    ]:
        for (h, k), radius in zip(indices, q_squared(indices, basis)):
            modes.append(
                {
                    "purpose": purpose,
                    "basis": basis,
                    "h": int(h),
                    "k": int(k),
                    "physical_q_squared": int(radius),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(modes)


def summarize(table):
    rows = []
    for label, subset in [("all", table), ("snr_selected", table[table.resolved])]:
        for metric in (
            "theta_deg",
            "c_AH",
            "c_B",
            "r_A",
            "w",
            "eta",
        ):
            values = subset[metric].to_numpy(float)
            absolute_floor = np.abs(subset.I_floor_A.to_numpy(float))
            floor = np.full_like(absolute_floor, np.nan)
            np.log10(absolute_floor, out=floor, where=absolute_floor > 0)
            valid = np.isfinite(values) & np.isfinite(floor)
            result = (
                spearmanr(floor[valid], values[valid])
                if valid.sum() >= 3
                and np.ptp(floor[valid]) > 0
                and np.ptp(values[valid]) > 0
                else (np.nan, np.nan)
            )
            rows.append(
                {
                    "population": label,
                    "metric": metric,
                    "n": len(subset),
                    "mean": float(np.mean(values)) if len(values) else np.nan,
                    "std_population": float(np.std(values)) if len(values) else np.nan,
                    "spearman_n": int(valid.sum()),
                    "floor_spearman_rho": float(result[0]),
                    "floor_spearman_p": float(result[1]),
                }
            )
    return pd.DataFrame(rows)
