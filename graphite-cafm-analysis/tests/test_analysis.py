import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from cafm import core
from cafm.coordinates import (
    convert_maps,
    convert_positions,
    convert_indices,
    direct_basis,
    reciprocal_basis,
    q_squared,
    site_weights,
    fourier_coefficients,
)
from cafm.preprocessing import flatten_lines_order1, prepare_map
from cafm.pipeline import PARAMETER_COLUMNS
from cafm.readout import (
    amplitude_snr,
    compute_readout,
    weighted_median,
    neighbors,
    SIGNAL_HK,
)
from run_analysis import main


class CoordinateTests(unittest.TestCase):
    def setUp(self):
        self.maps = np.random.default_rng(17).normal(size=(4, 64, 64))

    def test_basis_change_is_involution(self):
        np.testing.assert_array_equal(
            convert_maps(convert_maps(self.maps, "direct60"), "direct60"), self.maps
        )
        positions = np.array([[0.12, 0.23], [0.4, 0.7]])
        np.testing.assert_allclose(
            convert_positions(convert_positions(positions, "direct60"), "direct60"),
            positions,
        )

    def test_dual_vectors(self):
        for basis in ("direct120", "direct60"):
            np.testing.assert_allclose(
                reciprocal_basis(basis) @ direct_basis(basis),
                2 * np.pi * np.eye(2),
                atol=1e-14,
            )

    def test_physical_fourier_vectors(self):
        hk = np.array([[1, 0], [1, 1], [2, -1]])
        converted = convert_indices(hk, "direct60")
        np.testing.assert_allclose(
            hk @ reciprocal_basis("direct120"), converted @ reciprocal_basis("direct60")
        )
        np.testing.assert_array_equal(
            q_squared(hk, "direct120"), q_squared(converted, "direct60")
        )

    def test_pixel_centre_phase(self):
        hk = [[1, 0], [2, -1]]
        np.testing.assert_allclose(
            fourier_coefficients(self.maps, hk),
            fourier_coefficients(
                convert_maps(self.maps, "direct60"), convert_indices(hk, "direct60")
            ),
            atol=1e-15,
        )

    def test_kernel_follows_coordinates(self):
        position = [0.237, 0.619]
        first = site_weights((64, 64), position, "direct120")
        second = site_weights(
            (64, 64), convert_positions(position, "direct60"), "direct60"
        )
        np.testing.assert_allclose(first[::-1], second, atol=1e-16)
        np.testing.assert_allclose(
            first, core.periodic_gaussian_weight(64, position), atol=1e-16
        )

    def test_invalid_fourier_indices(self):
        with self.assertRaises(ValueError):
            convert_indices([[1, 0.5]], "direct120")
        with self.assertRaises(ValueError):
            fourier_coefficients(self.maps, [[0, 32]])


class PreprocessingTests(unittest.TestCase):
    def test_line_fit_against_independent_least_squares(self):
        image = np.random.default_rng(28).normal(size=(17, 23)) * 1e-11 + 4e-10
        original = image.copy()
        design = np.column_stack([np.arange(23), np.ones(23)])
        expected = np.stack(
            [
                row - design @ np.linalg.lstsq(design, row, rcond=None)[0]
                for row in image
            ]
        )
        actual = flatten_lines_order1(image)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=8e-25)
        np.testing.assert_array_equal(image, original)

    def test_line_offsets_and_slopes_removed(self):
        image = np.random.default_rng(29).normal(size=(19, 25))
        processed = flatten_lines_order1(image)
        np.testing.assert_allclose(processed.mean(axis=1), 0, atol=1e-15)
        np.testing.assert_allclose(processed @ (np.arange(25) - 12), 0, atol=1e-13)

    def test_column_axis(self):
        image = np.random.default_rng(30).normal(size=(19, 25))
        np.testing.assert_array_equal(
            flatten_lines_order1(image, 0), flatten_lines_order1(image.T).T
        )

    def test_processing_order_and_single_pass(self):
        image = np.random.default_rng(12).normal(size=(19, 25)) * 1e-11 + 4e-10
        expected = median_filter(flatten_lines_order1(image), size=3, mode="reflect")
        np.testing.assert_array_equal(prepare_map(image), expected)
        self.assertLess(abs(expected.mean()), 1e-11)

    def test_nine_pixel_median_edges(self):
        image = np.random.default_rng(31).normal(size=(7, 7))
        padded = np.pad(flatten_lines_order1(image), 1, mode="symmetric")
        expected = np.array(
            [
                [np.median(padded[j : j + 3, i : i + 3]) for i in range(7)]
                for j in range(7)
            ]
        )
        np.testing.assert_array_equal(prepare_map(image), expected)

    def test_invalid_line_input(self):
        for image, axis in [
            (np.ones(10), 1),
            (np.full((4, 4), np.nan), 1),
            (np.ones((4, 4)), 2),
        ]:
            with self.assertRaises(ValueError):
                flatten_lines_order1(image, axis)


class ReadoutTests(unittest.TestCase):
    def test_contrast_projections(self):
        values = core.contrast_coordinates(1, 2, 0)
        self.assertAlmostEqual(values["c_AH"], 3 / np.sqrt(6))
        self.assertAlmostEqual(values["c_B"], 1 / np.sqrt(2))
        self.assertAlmostEqual(values["theta_deg"], 30)

    def test_current_amplitude(self):
        self.assertAlmostEqual(core.contrast_amplitude(1, 2, 0), np.sqrt(2))
        self.assertEqual(core.contrast_amplitude(7, 7, 7), 0)
        self.assertAlmostEqual(core.contrast_amplitude(11, 12, 10), np.sqrt(2))

    def test_floor_is_raw_percentile(self):
        values = np.arange(25).reshape(5, 5) * 1e-12 + 4e-10
        self.assertEqual(core.current_floor(values), np.percentile(values, 5))

    def test_interpolated_weighted_median(self):
        self.assertEqual(weighted_median([1, 2, 3], [1, 1, 1]), 1.5)

    def test_leave_one_out_neighbors(self):
        for n in (2, 10, 51, 60):
            for i in range(n):
                indices, weights = neighbors(np.arange(n, dtype=float), i)
                self.assertNotIn(i, indices)
                self.assertEqual(len(indices), min(51, n - 1))
                self.assertAlmostEqual(weights.sum(), 1)

    def test_both_bases_preserve_readouts(self):
        maps = np.random.default_rng(31).normal(size=(60, 64, 64))
        theta = np.linspace(5, 75, len(maps))
        names = list(map(str, range(len(maps))))
        first, _ = compute_readout(maps, theta, names, "direct120")
        second, _ = compute_readout(
            convert_maps(maps, "direct60"), theta, names, "direct60"
        )
        for column in ("w", "eta", "snr"):
            np.testing.assert_allclose(first[column], second[column], atol=1e-12)
        np.testing.assert_array_equal(first.resolved, second.resolved)

    def test_snr_dc_gain_invariance(self):
        maps = np.random.default_rng(32).normal(size=(4, 64, 64))
        np.testing.assert_allclose(
            amplitude_snr(maps, "direct120")[2],
            amplitude_snr(maps * 3.7 + 21, "direct120")[2],
            atol=1e-12,
        )

    def test_selection_mode_radii(self):
        np.testing.assert_array_equal(
            np.sort(q_squared(SIGNAL_HK, "direct120")), [1, 1, 7, 7, 7, 7]
        )


class InputTests(unittest.TestCase):
    def test_text_units(self):
        self.assertAlmostEqual(core.parse_numeric_cell("3pA"), 3e-12, places=24)
        self.assertAlmostEqual(core.parse_numeric_cell("3 pA"), 3e-12, places=24)

    def test_only_confirmed_first_row_suffixes(self):
        frame = pd.DataFrame([[3e-10, "3e-10.1", "9e-10.1"], [1, "3e-10.1", 2]])
        actual, log = core.repair_duplicate_suffixes(frame)
        self.assertEqual(len(log), 1)
        self.assertEqual(actual.iat[0, 1], 3e-10)
        self.assertEqual(actual.iat[0, 2], "9e-10.1")
        self.assertEqual(actual.iat[1, 1], "3e-10.1")

    def test_output_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "input", Path(tmp) / "output"
            source.mkdir()
            output.mkdir()
            (output / "keep.txt").touch()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit
            ):
                main(["--input-dir", str(source), "--output-dir", str(output)])

    def test_cli_output_schema_and_basis_equivalence(self):
        y, x = np.indices((128, 128))
        p1 = 2 * np.pi * 12 * x / 128
        p2 = 2 * np.pi * (6 * x + 10 * y) / 128
        rng = np.random.default_rng(23)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "maps"
            source.mkdir()
            for i in range(3):
                image = (
                    4e-10
                    + 3e-11
                    * (
                        np.cos(p1 + 0.1 * i)
                        + 0.8 * np.cos(p2)
                        + 0.6 * np.cos(p2 - p1)
                        + 0.15 * np.cos(p1 + p2)
                        + 0.05 * rng.normal(size=x.shape)
                    )
                    + x * 1e-13
                )
                np.savetxt(source / f"{i}.csv", image, delimiter=",")
            for basis in ("direct120", "direct60"):
                with contextlib.redirect_stdout(io.StringIO()):
                    main(
                        [
                            "--input-dir",
                            str(source),
                            "--output-dir",
                            str(Path(tmp) / basis),
                            "--basis",
                            basis,
                        ]
                    )
            first = pd.read_csv(Path(tmp) / "direct120/parameters.csv")
            second = pd.read_csv(Path(tmp) / "direct60/parameters.csv")
            self.assertEqual(first.columns.tolist(), PARAMETER_COLUMNS)
            self.assertEqual(len(first), 3)
            for column in ("theta_deg", "w", "eta", "snr", "r_A"):
                np.testing.assert_allclose(first[column], second[column], atol=1e-12)
            self.assertTrue((first.resolved == (first.snr >= 5)).all())


if __name__ == "__main__":
    unittest.main()
