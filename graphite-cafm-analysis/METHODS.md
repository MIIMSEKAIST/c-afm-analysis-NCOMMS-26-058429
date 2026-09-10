# Calculation definitions

## Current maps

I_floor is the 5th percentile of the raw map. Each fast-scan line is fitted
by least squares to a*x+b using all pixels, and the fitted line is subtracted.
A square 3×3 median filter is applied once with reflect padding. No mask is
used and the original mean is not restored. This specifies the implementation;
pixel-level equivalence to an installed Igor/Asylum version has not been verified.

## Lattice and sites

FFT peak detection uses a plane-subtracted, Gaussian-high-pass, Hann-windowed
copy of the map. The selected reciprocal vectors define fractional coordinates
for averaging onto a 64×64 unit cell. Subtraction of the median and division by
1.4826×MAD are applied before and after folding. Three iterations of integer circular shifts align the cells.
The same lattice and shifts are applied to raw currents for the amplitude r.

Each normalized cell is searched over 32×32 origins and two atomic-offset pairs:
(1/3,2/3), (2/3,1/3) and (1/3,1/3), (2/3,2/3), with H at the origin.
B is the brighter atomic candidate. The score is the total B−H contrast, minus
five times the negative parts of B−A and A−H, plus 0.1 for strict B>A>H ordering.
A is then refined locally with B/H fixed. The radius is 0.045 on an 11×11 grid;
the score uses Gaussian widths 0.032/0.090, intensity weight 0.15, ordering
penalty 8 and displacement prior 0.020. Acceptance requires an improvement
of 1e−4 or restored ordering, with a maximum theta change of 25°.

Site averages use a periodic Gaussian of width 0.055. In direct120 coordinates
its kernel distance is min(du²+dv²+du*dv) over neighboring periodic images.
This is a fractional-coordinate kernel, not an isotropic Cartesian distance.

```text
c_AH = (I_A + I_B - 2 I_H) / sqrt(6)
c_B  = (I_B - I_A) / sqrt(2)
theta = atan2(abs(c_B), c_AH)
r = sqrt(sum_s (J_s - mean(J_A,J_B,J_H))²)
```

I_s are normalized site intensities; J_s are the raw site currents in amperes.
The output theta_deg is in degrees and r_A is in amperes.

## Fourier descriptors

The default direct120 basis uses q²=h²+hk+k². `--basis direct60` reverses the
second basis vector, map axis, site coordinates and Fourier indices together;
q² becomes h²−hk+k². Scalars are invariant under this coupled transformation.

Each image uses up to 51 other images nearest in theta. Their Gaussian weights
have width max(median(abs(delta_theta)),1°). Weighted medians are interpolated
at cumulative weight 0.5.

With L equal to the median log amplitude on q²=3 minus that on q²=1,
w = −0.5*(L − weighted_median(L_neighbors)). For eta, the weighted-median
neighbor log amplitudes are subtracted from each image's log amplitudes.
Weighted least squares over q²=1,3,4 fits the negative residual to
c + beta*(q²−1) + a*(gx²−gy²) + b*(2gx*gy); eta = sqrt(a²+b²).
Cartesian g is normalized by the first-shell vector magnitude.
Each shell has equal initial weight, multiplied by the reference amplitude
relative to its median and clipped to [0.2,5].

The SNR numerator is the mean amplitude at (−2,−1), (−1,−2), (−1,1),
(1,−1), (1,2), (2,1). Its denominator is the median amplitude at the twelve
integer pairs satisfying h²−hk+k²=7 in direct120 coordinates. The numerator
includes physical q²=1 and 7, so `snr` is not a pure sqrt(3)-shell SNR.
These index sets are transformed with the basis and recorded in fourier_modes.csv.
Selection is snr >= 5. Descriptors are calculated before selection; statistics
report both all maps and selected maps using log10(abs(I_floor)/A).
