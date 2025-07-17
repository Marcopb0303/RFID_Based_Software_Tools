"""Focused evaluation for 5-second FastGT experiment - simplified metrics output."""

from pathlib import Path
from typing_extensions import Annotated, Optional

import typer
import pandas as pd
import numpy as np
import json

try:
    import matplotlib
    matplotlib.use('Agg')  
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available - plots will be skipped")

from cartesian.stockroom.data.datapaths import DataPaths
from cartesian.stockroom.itx_data import load_itx
from cartesian.stockroom.maps.loader import load_neighbor_df
from cartesian.misc.logger import logger
from cartesian.stockroom.types.scan_type import Stockroom_Type

def load_associated_gt(csv_path: Path) -> pd.DataFrame:
    """Load the associated ground truth data."""
    df = pd.read_csv(csv_path, dtype={"BinLabel": str})
    df["epc"] = df["Tag Data"].str[4:]
    return df


def extract_back_neighbors(map_data: dict) -> dict[str, set[str]]:
    """Extract BACK neighbor relationships from map.json."""
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
                        back_neighbors[bin_label].add(neighbor_label)
                        if neighbor_label not in back_neighbors:
                            back_neighbors[neighbor_label] = set()
                        back_neighbors[neighbor_label].add(bin_label)# Bidirectional relationship
    
    return back_neighbors


def check_side_neighbors(bin1: str, bin2: str) -> bool:
    """Check if two bins are side neighbors using ±100 ITX."""
    try:
        return abs(int(bin1) - int(bin2)) == 100
    except (ValueError, TypeError):
        return False


def check_back_neighbors(bin1: str, bin2: str, back_neighbor_map: dict[str, set[str]]) -> bool:
    """Check if two bins are back neighbors using map data."""
    return bin2 in back_neighbor_map.get(bin1, set())


def analyze_neighbor_relationships(mutual_df: pd.DataFrame, neighbor_map_df: pd.DataFrame, 
                                  back_neighbor_map: dict[str, set[str]]) -> dict:
    """Analyze relationships between different neighbor tolerance methods."""
    
    total_predictions = len(mutual_df)
    
    # Calculate correctness for each method
    exact_matches = (mutual_df["BinLabel"] == mutual_df["BinLabel_itx"])
    
    # Neighbor Map (SIDE) correctness
    if neighbor_map_df is not None:
        neighbor_map_correct = (
            mutual_df.merge(
                neighbor_map_df,
                left_on=["BinLabel", "BinLabel_itx"],
                right_on=["BIN_1", "BIN_2"],
                how="left",
                indicator=True,
            )["_merge"] == "both"
        )
    else:
        neighbor_map_correct = pd.Series([False] * total_predictions)
    
    # 1NN ±100 correctness
    side_arithmetic_correct = pd.Series([False] * total_predictions)
    for idx, row in mutual_df.iterrows():
        if not exact_matches.iloc[idx]:  # Only check non-exact matches
            side_arithmetic_correct.iloc[idx] = check_side_neighbors(
                row["BinLabel"], row["BinLabel_itx"]
            )
    
    # BACK neighbor correctnessiterrows(
    back_correct = pd.Series([False] * total_predictions)
    for idx, row in mutual_df.iterrows():
        if not exact_matches.iloc[idx]:  # Only check non-exact matches
            back_correct.iloc[idx] = check_back_neighbors(
                row["BinLabel"], row["BinLabel_itx"], back_neighbor_map
            )
    
    # Combined method correctness
    top_0s_total = exact_matches
    neighbor_map_total = exact_matches | neighbor_map_correct
    one_nn_100_total = exact_matches | side_arithmetic_correct
    one_nn_back_total = exact_matches | side_arithmetic_correct | back_correct
    
    # Analysis of overlaps
    analysis = {
        'total_predictions': total_predictions,
        'exact_matches': exact_matches.sum(),
        'neighbor_map_additional': neighbor_map_correct.sum(),
        'side_arithmetic_additional': side_arithmetic_correct.sum(),
        'back_additional': back_correct.sum(),
        
        # Total correct for each method
        'top_0s_correct': top_0s_total.sum(),
        'neighbor_map_correct': neighbor_map_total.sum(),
        'one_nn_100_correct': one_nn_100_total.sum(),
        'one_nn_back_correct': one_nn_back_total.sum(),
        
        # Overlap analysis
        'back_also_in_100': (side_arithmetic_correct & back_correct).sum(),
        'back_only': (back_correct & ~side_arithmetic_correct).sum(),
        'side_map_also_in_100': (neighbor_map_correct & side_arithmetic_correct).sum(),
        'side_map_only': (neighbor_map_correct & ~side_arithmetic_correct).sum(),
        'hundred_only': (side_arithmetic_correct & ~neighbor_map_correct).sum(),
    }
    
    return analysis


def create_correctness_plot(analysis: dict, dataset_name: str, output_dir: Path) -> None:
    """Create a bar plot showing percentage of correct predictions for each method."""
    
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available - skipping plot generation")
        return
    
    total = analysis['total_predictions']
    methods = ['Top-0s', 'Neighbor Map', '1NN ±100', '1NN ±100 + BACK']
    correct_counts = [
        analysis['top_0s_correct'],
        analysis['neighbor_map_correct'], 
        analysis['one_nn_100_correct'],
        analysis['one_nn_back_correct']
    ]
    
    percentages = [count / total * 100 for count in correct_counts]
    
    try:
        # Create the plot
        logger.info(f"Creating plot with {len(methods)} methods, total predictions: {total}")
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, percentages, 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                       alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add percentage labels on bars
        for bar, pct, count in zip(bars, percentages, correct_counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{pct:.1f}%\n({count}/{total})',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'Prediction Correctness by Evaluation Method\n{dataset_name}', 
                  fontsize=14, fontweight='bold')
        plt.ylabel('Percentage of Correct Predictions (%)', fontsize=12)
        plt.xlabel('Evaluation Method', fontsize=12)
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        
        # Improve layout
        plt.tight_layout()
        
        # Save the plot
        plot_path = output_dir / f"correctness_comparison_{dataset_name}.png"
        logger.info(f"Saving plot to: {plot_path}")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Correctness plot saved to: {plot_path}")
        
        # Verify file was created
        if plot_path.exists():
            file_size = plot_path.stat().st_size
            logger.info(f"Plot file created successfully: {file_size} bytes")
        else:
            logger.error(f"Plot file was not created at {plot_path}")
    
    except Exception as e:
        logger.error(f"Failed to create plot: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        logger.info("Plot generation failed, but analysis continues...")


def print_detailed_analysis(analysis: dict, dataset_name: str) -> None:
    """Print detailed analysis of neighbor relationships."""
    
    total = analysis['total_predictions']
    
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS: {dataset_name}")
    print(f"{'='*60}")
    
    print(f"\n OVERALL CORRECTNESS SUMMARY:")
    print(f"   Total predictions analyzed: {total:,}")
    print(f"   Top-0s (exact):             {analysis['top_0s_correct']:,} ({analysis['top_0s_correct']/total*100:.1f}%)")
    print(f"   Neighbor Map (SIDE):        {analysis['neighbor_map_correct']:,} ({analysis['neighbor_map_correct']/total*100:.1f}%)")
    print(f"   1NN ±100 (ITX):             {analysis['one_nn_100_correct']:,} ({analysis['one_nn_100_correct']/total*100:.1f}%)")
    print(f"   1NN ±100 + BACK:            {analysis['one_nn_back_correct']:,} ({analysis['one_nn_back_correct']/total*100:.1f}%)")
    
    print(f"\n NEIGHBOR TOLERANCE BREAKDOWN:")
    print(f"   Exact matches:              {analysis['exact_matches']:,}")
    print(f"   Additional via Neighbor Map: {analysis['neighbor_map_additional']:,}")
    print(f"   Additional via 1NN ±100:    {analysis['side_arithmetic_additional']:,}")
    print(f"   Additional via BACK:        {analysis['back_additional']:,}")
    
    print(f"\n KEY RESEARCH QUESTIONS:")
    
    # Question 1: What percentage of 1NN + BACK are already inside 1NN?
    if analysis['one_nn_back_correct'] > analysis['one_nn_100_correct']:
        back_additional_benefit = analysis['one_nn_back_correct'] - analysis['one_nn_100_correct']
        back_overlap_pct = (analysis['back_also_in_100'] / analysis['back_additional']) * 100 if analysis['back_additional'] > 0 else 0
        print(f"   What % of BACK benefits are already in 1NN ±100?")
        print(f"       → {back_overlap_pct:.1f}% of BACK relationships are also ±100")
        print(f"       → BACK adds {back_additional_benefit} additional correct predictions")
    else:
        print(f"   What % of BACK benefits are already in 1NN ±100?")
        print(f"       → 100% - All BACK relationships are covered by ±100 arithmetic")
    
    # Question 2: What percentage of SIDE (Map neighbors) match with arithmetic 1NN?
    if analysis['neighbor_map_additional'] > 0:
        side_map_overlap_pct = (analysis['side_map_also_in_100'] / analysis['neighbor_map_additional']) * 100
        print(f"   What % of SIDE (Map) benefits match ±100 arithmetic?")
        print(f"       → {side_map_overlap_pct:.1f}% of SIDE relationships are also ±100")
        print(f"       → {analysis['side_map_only']} SIDE relationships are unique (not ±100)")
        print(f"       → {analysis['hundred_only']} ±100 relationships are unique (not SIDE)")
    else:
        print(f"   What % of SIDE (Map) benefits match ±100 arithmetic?")
        print(f"       → No additional SIDE relationships found")
    
    # Question 3: Performance ranking
    methods_performance = [
        ("Top-0s", analysis['top_0s_correct']),
        ("Neighbor Map", analysis['neighbor_map_correct']),
        ("1NN ±100", analysis['one_nn_100_correct']),
        ("1NN ±100 + BACK", analysis['one_nn_back_correct'])
    ]
    methods_performance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   Q3: Performance ranking for this dataset:")
    for i, (method, correct) in enumerate(methods_performance, 1):
        pct = correct / total * 100
        print(f"       {i}. {method}: {correct:,} correct ({pct:.1f}%)")
    
    print(f"\n INSIGHTS FOR 5-SECOND EXPERIMENT:")
    best_method = methods_performance[0][0]
    print(f"   • Best performing method: {best_method}")
    print(f"   • Spatial vs Arithmetic: {'SIDE spatial' if analysis['neighbor_map_correct'] > analysis['one_nn_100_correct'] else 'Arithmetic ±100'} performs better")
    print(f"   • BACK neighbor value: {'Minimal' if analysis['one_nn_back_correct'] == analysis['one_nn_100_correct'] else 'Significant'} additional benefit")


def main(
    target_sections: Annotated[
        list[Stockroom_Type],
        typer.Option(
            ...,
            "-s",
            "--section",
            help="Stockroom sections targeted by the GT scan. Options are WOMAN, MAN, KIDS.",
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
            help="Path to JSON file specifying the map floor.",
        ),
    ] = None,
    dataset_name: Annotated[
        Optional[str],
        typer.Option(
            help="Name of the dataset to evaluate. If provided, paths will be inferred.",
        ),
    ] = None,
    save_plot: Annotated[
        bool,
        typer.Option(
            "--plot",
            "-p",
            help="Save correctness comparison plot.",
            is_flag=True,
        ),
    ] = True,
) -> None:
    """Focused evaluation for 5-second FastGT experiment."""
    
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

    logger.info(f"Starting focused evaluation for: {dataset_name or 'custom dataset'}")

    # Load data
    associated_df = load_associated_gt(associated_gt_path)
    itx_df = load_itx(itx_path, target_sections)
    neighbor_map_df = (
        load_neighbor_df(neighbor_map_path).query("type == 'SIDE'")
        if neighbor_map_path and neighbor_map_path.exists()
        else None
    )

    # Load BACK neighbors
    back_neighbor_map = {}
    if neighbor_map_path and neighbor_map_path.exists():
        with open(neighbor_map_path, 'r') as f:
            map_data = json.load(f)
        back_neighbor_map = extract_back_neighbors(map_data)

    # Prepare data for evaluation (same logic as original)
    itx_df = itx_df.copy().query("ZoneLocation == 'Stockroom'")
    itx_df = itx_df[itx_df["BinLabel"].isin(associated_df["BinLabel"])]
    associated_df = associated_df.query("not BinLabel.str.startswith('C')")

    mutual_df = associated_df.merge(
        itx_df, left_on="epc", right_index=True, suffixes=("", "_itx"), how="right"
    ).reset_index(drop=True)

    logger.info(f"Analyzing {len(mutual_df)} mutual EPC records")

    # Perform detailed analysis
    analysis = analyze_neighbor_relationships(mutual_df, neighbor_map_df, back_neighbor_map)
    
    # Print detailed analysis
    print_detailed_analysis(analysis, dataset_name or "Custom Dataset")
    
    # Create and save plot if requested
    if save_plot:
        output_dir = associated_gt_path.parent if associated_gt_path else Path(".")
        logger.info(f"Attempting to create plot in directory: {output_dir}")
        create_correctness_plot(analysis, dataset_name or "custom", output_dir)
    
    # Save analysis results as JSON
    output_dir = associated_gt_path.parent if associated_gt_path else Path(".")
    analysis_path = output_dir / f"correctness_analysis_{dataset_name or 'custom'}.json"
    
    # Convert any non-serializable types
    analysis_serializable = {k: int(v) if isinstance(v, (np.int64, np.int32)) else v 
                           for k, v in analysis.items()}
    
    with open(analysis_path, 'w') as f:
        json.dump(analysis_serializable, f, indent=2)
    
    logger.info(f"Analysis results saved to: {analysis_path}")
    logger.info("Focused evaluation completed!")


if __name__ == "__main__":
    typer.run(main)