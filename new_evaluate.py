"""Functions to evaluate fastgt w.r.t. ITX data."""

from pathlib import Path
from typing_extensions import Annotated, Optional

import typer
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import json

from cartesian.stockroom.data.datapaths import DataPaths
from cartesian.stockroom.itx_data import load_itx
from cartesian.metrics.metrics import precision_recall
from cartesian.stockroom.maps.loader import load_neighbor_df
from cartesian.misc.logger import logger
from cartesian.stockroom.types.scan_type import Stockroom_Type


def load_associated_gt(csv_path: Path) -> pd.DataFrame:
    """Load the associated ground truth data.

    This is generally created from the `associate_gt.py` script.

    Args:
        csv_path: Path to the associated ground truth data.

    Returns:
        The dataframe with the associated ground truth data.
    """
    df = pd.read_csv(csv_path, dtype={"BinLabel": str})

    # The Tag Data column actually contains EPC+PC
    # so we need to remove the PC (first 4 chars)
    df["epc"] = df["Tag Data"].str[4:]

    return df


def extract_back_neighbors(map_data: dict) -> dict[str, set[str]]:
    """Extract BACK neighbor relationships from map.json.

    Args:
        map_data: Dictionary containing the map data loaded from JSON.

    Returns:
        Dictionary mapping bin labels to sets of their BACK neighbors.
    """
    back_neighbors = {}

    for rack in map_data.get("racks", []):
        for bin_info in rack.get("bins", []):
            bin_label = bin_info.get("label")
            if not bin_label:
                continue

            if bin_label not in back_neighbors:
                back_neighbors[bin_label] = set()

            for neighbor_info in bin_info.get("neighbors", []):
                if neighbor_info.get("type") == "BACK":
                    neighbor_label = neighbor_info.get("neighbor")
                    if neighbor_label:
                        # Add bidirectional relationship
                        back_neighbors[bin_label].add(neighbor_label)
                        if neighbor_label not in back_neighbors:
                            back_neighbors[neighbor_label] = set()
                        back_neighbors[neighbor_label].add(bin_label)

    return back_neighbors


def check_side_neighbors(bin1: str, bin2: str) -> bool:
    """Check if two bins are side neighbors using ±100 arithmetic.

    Args:
        bin1: First bin label.
        bin2: Second bin label.

    Returns:
        True if bins are side neighbors (±100 apart), False otherwise.
    """
    try:
        return abs(int(bin1) - int(bin2)) == 100
    except (ValueError, TypeError):
        return False


def check_back_neighbors(bin1: str, bin2: str, back_neighbor_map: dict[str, set[str]]) -> bool:
    """Check if two bins are back neighbors using map data.

    Args:
        bin1: First bin label.
        bin2: Second bin label.
        back_neighbor_map: Dictionary of BACK neighbor relationships.

    Returns:
        True if bins are back neighbors, False otherwise.
    """
    return bin2 in back_neighbor_map.get(bin1, set())


def evaluate(
    associated_df: pd.DataFrame,
    itx_df: pd.DataFrame,
    filter_bins_eval: list[str] | None = None,
    neighbor_map_df: pd.DataFrame | None = None,
    neighbor_itx: bool = False,
    back_neighbor_map: dict[str, set[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute precision/recall of the associated ground truth data against ITX data.

    The evaluation filters unlabelled bins.

    If `filter_bins_eval` is provided, consider any EPCs that either ITX or
    the algorithm assigned to those bins. This considers the case where EPCs in the
    buffer zone are assigned to the evaluation bins, as those are counted as false
    positives.

    Args:
        associated_df: The associated ground truth data.
        itx_df: The ITX data.
        filter_bins_eval: If provided, only consider these bins for evaluation.
        neighbor_map_df: If provided, accepts allocations to neighboring bins as
                         correct. This df must have the topology of the neighbor bins.
        neighbor_itx: If True, considers bins that are within +- 100 of the ITX bin
                        label as correct. This is useful for ITX data that does not
                        have a neighbor map. This is ignored if `neighbor_map_df`
                        is provided.
        back_neighbor_map: If provided, accepts allocations to BACK neighbors as
                         correct when combined with neighbor_itx for 1NN + BACK method.

    Returns:
        A tuple of (precision, recall, thresholds, avg_prec).

    Raises:
        ValueError: If both `neighbor_map_df` and `neighbor_itx` are provided without
                   back_neighbor_map, or if conflicting neighbor parameters are used.
    """
    # Only include stockroom items
    itx_df = itx_df.copy().query("ZoneLocation == 'Stockroom'")

    # Only include bins that we have labeled during GT collection
    itx_df = itx_df[itx_df["BinLabel"].isin(associated_df["BinLabel"])]

    # Remove EPCs from "unlabeled" bins
    # These are removed because we cannot ground-truth them via ITX data
    associated_df = associated_df.query("not BinLabel.str.startswith('C')")

    # For evaluation we only consider the EPCs that exist in ITX data
    # because the other EPCs are unallocated items which we cannot ground-truth via ITX
    mutual_df = associated_df.merge(
        itx_df, left_on="epc", right_index=True, suffixes=("", "_itx"), how="right"
    ).reset_index(drop=True)

    # If evaluating in specific bins, considers all the items that either our algorithm
    # or ITX assigned to those bins, and compute Precision/Recall for those EPCs
    if filter_bins_eval is not None:
        mutual_df = mutual_df.query(
            "BinLabel in @filter_bins_eval or BinLabel_itx in @filter_bins_eval"
        )

    # Validate parameter combinations
    if neighbor_map_df is not None and neighbor_itx and back_neighbor_map is None:
        raise ValueError(
            "Cannot use both neighbor_map_df and neighbor_itx at the same time."
        )

    correct_associations = mutual_df["BinLabel"] == mutual_df["BinLabel_itx"]

    # If neighbor_map_df is provided, consider neighbor acceptable bin allocations
    if neighbor_map_df is not None:
        # Use merge for to find out if estimated and ITX bins are neighbors
        correct_associations |= (
            mutual_df.merge(
                neighbor_map_df,
                left_on=["BinLabel", "BinLabel_itx"],
                right_on=["BIN_1", "BIN_2"],
                how="left",
                indicator=True,
            )["_merge"]
            == "both"
        )
    elif neighbor_itx and back_neighbor_map is None:
        # Original ±100 logic for ITX 1NN method
        correct_associations |= mutual_df.apply(
            lambda row: (
                (
                    row["BinLabel_itx"] != ""
                    and not pd.isna(row["BinLabel_itx"])
                    and not pd.isna(row["BinLabel"])
                )
                and (
                    row["BinLabel"] == row["BinLabel_itx"]
                    or abs(int(row["BinLabel"]) - int(row["BinLabel_itx"])) == 100
                )
            ),
            axis=1,
        )
    elif neighbor_itx and back_neighbor_map is not None:
        # NEW: 1NN + BACK method - combines ±100 arithmetic with BACK spatial neighbors
        correct_associations |= mutual_df.apply(
            lambda row: (
                (
                    row["BinLabel_itx"] != ""
                    and not pd.isna(row["BinLabel_itx"])
                    and not pd.isna(row["BinLabel"])
                )
                and (
                    row["BinLabel"] == row["BinLabel_itx"]
                    or check_side_neighbors(row["BinLabel"], row["BinLabel_itx"])
                    or check_back_neighbors(row["BinLabel"], row["BinLabel_itx"], back_neighbor_map)
                )
            ),
            axis=1,
        )

    # compute precision and recall
    precision, recall, thresholds, avg_prec = precision_recall(
        confidence=mutual_df["confidence"].to_numpy(),
        tp=correct_associations.to_numpy(),
    )

    return precision, recall, thresholds, avg_prec


def main(
    target_sections: Annotated[
        list[Stockroom_Type],
        typer.Option(
            ...,
            "-s",
            "--section",
            help=(
                "Stockroom sections targeted by the GT scan. Options are WOMAN, MAN, KIDS."  # noqa: E501
            ),
        ),
    ],
    associated_gt_path: Annotated[
        Optional[Path],
        typer.Argument(help="Path to the associated ground truth csv."),
    ] = None,
    itx_path: Annotated[
        Optional[Path],
        typer.Argument(help="Path to itx_data.csv file."),
    ] = None,
    neighbor_map_path: Annotated[
        Optional[Path],
        typer.Option(
            help=(
                "This must be a JSON specifying the map floor. "
                "If provided, accepts allocations to neighboring bins as correct."
            )
        ),
    ] = None,
    dataset_name: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Name of the dataset to evaluate. If provided, paths will be inferred "
                "based on this name."
            )
        ),
    ] = None,
    html_output: Annotated[
        bool,
        typer.Option(
            "--html",
            "-h",
            help="Enable saving of the summary HTML.",
            is_flag=True,
        ),
    ] = True,
    debug_mode: Annotated[
        bool,
        typer.Option(
            "--debug",
            "-d",
            help="Enable debug mode to save the metrics as JSON outputs.",
            is_flag=True,
        ),
    ] = False,
) -> None:
    """Evaluates quality of a stockroom association by comparing against ITX data."""
    if dataset_name:
        data_path = Path("data")
        itx_path = DataPaths.itx_data_path(data_path, dataset_name)
        associated_gt_path = (
            DataPaths.raw_dir_path(data_path, dataset_name) / "associated_gt.csv"
        )
        neighbor_map_path = DataPaths.map_path(data_path, dataset_name)

    if not associated_gt_path or not itx_path:
        raise ValueError(
            "Either 'dataset_name' must be provided or both 'associated_gt_path' and "
            "'itx_path' must be specified."
        )

    associated_df = load_associated_gt(associated_gt_path)
    itx_df = load_itx(itx_path, target_sections)
    neighbor_map_df = (
        load_neighbor_df(neighbor_map_path).query("type == 'SIDE'")
        if neighbor_map_path
        else None
    )

    # Load map data and extract BACK neighbors for the 4th evaluation method
    back_neighbor_map = None
    if neighbor_map_path:
        with open(neighbor_map_path, 'r') as f:
            map_data = json.load(f)
        back_neighbor_map = extract_back_neighbors(map_data)

    summary_str = (
        "#EPCs not found in ITX data: "
        f"{(~associated_df['epc'].isin(itx_df.index)).sum()}\n"
        "#EPCs not found in collected data: "
        f"{(~itx_df.index.isin(associated_df['epc'])).sum()}\n"
        "#EPCs found in both: "
        f"{associated_df['epc'].isin(itx_df.index).sum()}"
    )
    logger.info(summary_str)

    if debug_mode:
        results = {}
    if html_output:
        fig = go.Figure()

    # Extended evaluation loop with 4th method
    for neigh_map, neigh_itx, back_map, label in (
        (None, False, None, "Top-0s"),
        (neighbor_map_df, False, None, "Neighbor Map"),
        (None, True, None, "Neighbor 100s"),
        (None, True, back_neighbor_map, "1NN + BACK"),  # NEW 4th method
    ):
        if neighbor_map_df is None and label == "Neighbor Map":
            logger.warning(
                "Neighbor Map is not provided, skipping evaluation for Neighbor Map."
            )
            continue
        if back_neighbor_map is None and label == "1NN + BACK":
            logger.warning(
                "Map data is not provided, skipping evaluation for 1NN + BACK."
            )
            continue

        precision, recall, thresholds, avg_prec = evaluate(
            associated_df=associated_df,
            itx_df=itx_df,
            neighbor_map_df=neigh_map,
            neighbor_itx=neigh_itx,
            back_neighbor_map=back_map,
        )
        logger.info(f"{label}")
        logger.info(f"Overall Precision: {precision[0]:.3f}")
        logger.info(f"Overall Recall: {recall[0]:.3f}")
        logger.info(f"Average Precision: {avg_prec:.3f}")
        if debug_mode:
            results[label] = {
                "overall_precision": precision[0].astype(float),
                "overall_recall": recall[0].astype(float),
                "average_precision": float(avg_prec),
                "precision": precision.astype(float).round(4).tolist(),
                "recall": recall.astype(float).round(4).tolist(),
            }
        if html_output:
            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    text=thresholds,
                    mode="lines",
                    hoverinfo="x+y+text",
                    name=label,
                )
            )

    if debug_mode:
        json.dump(
            results,
            open(associated_gt_path.with_name("report_gt.json"), "w"),
            indent=4,
        )
    if html_output:
        fig.update_layout(
            title=f"{associated_gt_path.parent.name} - {itx_path.name}",
            xaxis_title="Recall",
            yaxis_title="Precision",
        )
        fig.write_html(associated_gt_path.with_name("report_gt.html"))


if __name__ == "__main__":
    typer.run(main)
