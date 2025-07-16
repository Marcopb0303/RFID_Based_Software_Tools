import pandas as pd
import numpy as np
from pathlib import Path

script_dir = Path(__file__).parent  # fastgt directory
cartesian_root = script_dir.parent.parent.parent  # cartesian-tools directory
data_file = cartesian_root / "data/multi-section/data/14766_PlazaEspana_20250327T110257Z_000/TagReadings.csv"

print(f"Looking for file at: {data_file}")

if not data_file.exists():
    print("File not found! Let's check what datasets are available:")
    data_dir = cartesian_root / "data/multi-section/data"
    if data_dir.exists():
        datasets = list(data_dir.iterdir())
        print("Available datasets:")
        for dataset in datasets:
            if dataset.is_dir():
                print(f"  {dataset.name}")
                tagreadings = dataset / "TagReadings.csv"
                if tagreadings.exists():
                    print(f"    TagReadings.csv found")
                else:
                    print(f"    TagReadings.csv missing")
    exit(1)

df = pd.read_csv(data_file)

print("RSSI Distribution Analysis:")
print(f"Min RSSI: {df['RSSI'].min()}")
print(f"Max RSSI: {df['RSSI'].max()}")
print(f"Range: {df['RSSI'].max() - df['RSSI'].min()}")
print(f"Mean: {df['RSSI'].mean():.2f}")
print(f"Std: {df['RSSI'].std():.2f}")

print(f"\nQuantiles:")
for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
    print(f"{q*100:4.0f}%: {df['RSSI'].quantile(q):6.2f}")

print(f"\nShape assessment:")
print(f"Skewness: {df['RSSI'].skew():.2f}")  # >0 = right-skewed, <0 = left-skewed
print(f"Total readings: {len(df):,}")
