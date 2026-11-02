"""Check the consistency of the saved project results without running TERRA.

Run from the repository root: python scripts/verify_results.py
Requires numpy, pandas and scipy. The original expression matrix is not included,
so this script cannot validate the model predictions or recalculate marker scores.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MARKERS = {
    "IRF1", "IFIT1", "IFIT2", "IFIT3", "MX1",
    "CXCL9", "CXCL10", "CXCL11", "TAP1",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_finite(frame, columns):
    require(
        np.isfinite(frame[columns].to_numpy(dtype=float)).all(),
        f"Non-finite values in {columns}",
    )


def compare_columns(left, right, columns, description):
    """Compare tables by cell ID without relying on row order."""
    left = left.set_index("cell_id")
    right = right.set_index("cell_id")
    require(left.index.is_unique and right.index.is_unique, "Duplicate cell IDs")
    require(set(left.index) <= set(right.index), f"Missing IDs in {description}")
    for column in columns:
        np.testing.assert_allclose(
            left[column].to_numpy(dtype=float),
            right.loc[left.index, column].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Mismatch in {description}: {column}",
        )


def main():
    core = pd.read_csv(RESULTS / "ifn_core_scores.csv")
    model100 = pd.read_csv(RESULTS / "ifn_stat1_100.csv")
    model60 = pd.read_csv(RESULTS / "ifn_stat1_tdrd3_60.csv")
    shifts100 = pd.read_csv(RESULTS / "stat1_100_cells.csv")
    shifts60 = pd.read_csv(RESULTS / "stat1_vs_reference.csv")
    markers = pd.read_csv(RESULTS / "ifn_gene_detection.csv")
    summary = pd.read_csv(RESULTS / "ifn_analysis_summary.csv")
    leave_one_out = pd.read_csv(
        RESULTS / "ifn_leave_one_gene_out.csv", keep_default_na=False
    )

    for label, frame, expected_count in [
        ("core", core, 1881), ("model100", model100, 100),
        ("model60", model60, 60), ("shifts100", shifts100, 100),
        ("shifts60", shifts60, 60),
    ]:
        require(len(frame) == expected_count, f"Unexpected row count in {label}")
        require(frame.cell_id.notna().all() and frame.cell_id.is_unique,
                f"Missing or duplicate cell IDs in {label}")
        check_finite(frame, frame.select_dtypes(include="number").columns)

    require(set(model60.cell_id) <= set(model100.cell_id), "60 cells not in the 100")
    require(set(model100.cell_id) == set(shifts100.cell_id), "100-cell ID mismatch")
    require(set(model60.cell_id) == set(shifts60.cell_id), "60-cell ID mismatch")
    core_columns = ["x", "y", "total_counts", "STAT1_counts", "STAT1_log1p_10k",
                    "IFN_marker_score", "IFN_markers_detected"]
    compare_columns(model100, core, core_columns, "100 cells vs core")
    compare_columns(model60, core, core_columns, "60 cells vs core")
    compare_columns(model100, shifts100, ["x", "y", "STAT1_KO_shift"],
                    "100-cell results")
    compare_columns(model60, shifts60, ["STAT1_shift", "reference_shift", "difference"],
                    "60-cell results")
    require((model100.STAT1_counts > 0).all(), "Unexpected STAT1-negative selected cell")
    require((shifts60.reference_gene == "ENSG00000083544").all(),
            "Unexpected comparison gene")
    require((model100.STAT1_KO_shift >= 0).all(), "Negative embedding distance")
    require(((model60.STAT1_shift >= 0) & (model60.reference_shift >= 0)).all(),
            "Negative matched embedding distance")
    # The separately written float32 shifts and difference have CSV rounding.
    np.testing.assert_allclose(model60.difference,
                               model60.STAT1_shift - model60.reference_shift,
                               rtol=1e-6, atol=5e-8)

    require((core.total_counts > 0).all(), "Nonpositive total count")
    np.testing.assert_allclose(core.STAT1_log1p_10k,
                               np.log1p(core.STAT1_counts * 10000 / core.total_counts),
                               rtol=1e-6, atol=1e-7)
    require(set(markers.gene) == MARKERS and len(markers) == len(MARKERS),
            "Unexpected marker list")
    np.testing.assert_allclose(markers.percent_detected,
                               markers.cells_detected * 100 / len(core),
                               rtol=1e-10, atol=1e-12)
    require(markers.cells_detected.sum() == core.IFN_markers_detected.sum(),
            "Marker detection totals do not agree")
    require(((core.IFN_marker_score == 0) == (core.IFN_markers_detected == 0)).all(),
            "Marker zero-score and detection counts disagree")
    require((core.IFN_marker_score == 0).sum() == 810, "Unexpected zero-score count")

    comparisons = [
        ("Observed STAT1 expression vs IFN score", core, "STAT1_log1p_10k"),
        ("STAT1 embedding shift vs IFN score", model100, "STAT1_KO_shift"),
        ("Matched STAT1 shift vs IFN score", model60, "STAT1_shift"),
        ("Matched TDRD3 shift vs IFN score", model60, "reference_shift"),
        ("Matched shift difference vs IFN score", model60, "difference"),
        ("Total counts vs IFN score: technical check", model100, "total_counts"),
        ("STAT1 token rank vs IFN score: technical check", model100, "STAT1_token_rank"),
    ]
    require(summary.comparison.is_unique and len(summary) == len(comparisons),
            "Unexpected correlation table")
    summary = summary.set_index("comparison")
    for label, frame, column in comparisons:
        require(label in summary.index and summary.loc[label, "n_cells"] == len(frame),
                f"Unexpected correlation row: {label}")
        rho = float(spearmanr(frame.IFN_marker_score, frame[column]).statistic)
        np.testing.assert_allclose(rho, summary.loc[label, "spearman_rho"],
                                   rtol=1e-10, atol=1e-12, err_msg=label)

    require(len(leave_one_out) == 10 and
            set(leave_one_out.excluded_gene) == MARKERS | {"none"},
            "Unexpected leave-one-out table")
    full = leave_one_out.set_index("excluded_gene").loc["none"]
    np.testing.assert_allclose(full.zero_score_percent_core,
                               100 * (core.IFN_marker_score == 0).mean(), atol=1e-12)
    for column, label in [
        ("rho_observed_STAT1_core", comparisons[0][0]),
        ("rho_STAT1_shift_100", comparisons[1][0]),
        ("rho_shift_difference_60", comparisons[4][0]),
    ]:
        np.testing.assert_allclose(full[column], summary.loc[label, "spearman_rho"],
                                   rtol=1e-10, atol=1e-12)

    for name in ["stat1_100_cells.png", "stat1_vs_reference.png", "ifn_context_analysis.png"]:
        require((RESULTS / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n",
                f"Missing or invalid PNG: {name}")

    print("PASS: cell IDs, saved values, marker detection, and all seven correlations")
    print("PASS: leave-one-out table structure and the full-score baseline row")
    print(f"Core cells: {len(core)}; model cells: {len(model100)}; paired cells: {len(model60)}")
    print(f"Zero IFN scores: {(core.IFN_marker_score == 0).sum()} / {len(core)}")
    print("Note: this does not rerun TERRA or reconstruct scores from the raw data.")


if __name__ == "__main__":
    main()
