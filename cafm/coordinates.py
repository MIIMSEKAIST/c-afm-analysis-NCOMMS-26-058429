"""Direct/reciprocal coordinates and periodic Fourier coefficients."""

import numpy as np

BASES = ("direct120", "direct60")


def basis_sign(basis):
    if basis not in BASES:
        raise ValueError(f"Unknown basis: {basis}")
    return 1 if basis == "direct120" else -1


def convert_maps(maps, basis):
    sign = basis_sign(basis)
    values = np.asarray(maps, dtype=float)
    if values.ndim < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite two-dimensional maps")
    return values.copy() if sign == 1 else np.flip(values, axis=-2).copy()


def convert_positions(uv, basis):
    uv = np.asarray(uv, dtype=float)
    if uv.shape[-1] != 2 or not np.isfinite(uv).all():
        raise ValueError("Expected finite (u, v) coordinates")
    return uv * np.array([1, basis_sign(basis)]) % 1


def convert_indices(hk, basis):
    values = np.asarray(hk)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or (not np.isfinite(values).all())
        or (not np.equal(values, np.round(values)).all())
    ):
        raise ValueError("Expected integer (h, k) pairs")
    return values.astype(int) * np.array([1, basis_sign(basis)])


def direct_basis(basis):
    return np.array([[1.0, -0.5], [0.0, np.sqrt(3) / 2]]) @ np.diag(
        [1.0, basis_sign(basis)]
    )


def reciprocal_basis(basis):
    return 2 * np.pi * np.linalg.inv(direct_basis(basis))


def q_squared(hk, basis):
    hk = np.asarray(hk)
    h, k = (hk[..., 0], hk[..., 1])
    return h * h + basis_sign(basis) * h * k + k * k


def reciprocal_cartesian(hk, basis):
    hk = np.asarray(hk)
    h, k = (hk[..., 0], hk[..., 1])
    return np.stack(
        [h, (h + 2 * basis_sign(basis) * k) / np.sqrt(3)], axis=-1
    ) / np.sqrt(4 / 3)


def site_weights(shape, position, basis, sigma=0.055):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    ny, nx = shape
    u, v = np.meshgrid((np.arange(nx) + 0.5) / nx, (np.arange(ny) + 0.5) / ny)
    position = np.asarray(position, dtype=float)
    if position.shape != (2,) or not np.isfinite(position).all():
        raise ValueError("Expected one finite site coordinate")
    position = position % 1
    distance = np.full(shape, np.inf)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            du, dv = (u - position[0] + i, v - position[1] + j)
            distance = np.minimum(
                distance, du * du + dv * dv + basis_sign(basis) * du * dv
            )
    weights = np.exp(-distance / (2 * sigma * sigma))
    return weights / weights.sum()


def fourier_coefficients(maps, hk):
    values = np.asarray(maps, dtype=float)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Expected finite maps with shape (images, rows, columns)")
    indices = convert_indices(hk, "direct120")
    h, k = indices.T
    ny, nx = values.shape[-2:]
    if np.any(np.abs(h) >= nx / 2) or np.any(np.abs(k) >= ny / 2):
        raise ValueError("Requested modes must lie strictly below Nyquist")
    values = values - values.mean(axis=(-2, -1), keepdims=True)
    ft = np.fft.fft2(values, axes=(-2, -1)) / (ny * nx)
    return ft[:, k % ny, h % nx] * np.exp(-1j * np.pi * (h / nx + k / ny))
