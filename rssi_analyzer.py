#!/usr/bin/env python3
"""
Simple RSSI Analysis Tool
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class SimpleRSSIAnalyzer:
    def __init__(self):
        """Set up the analyzer - find where our data files are"""
        self.data_folder = "../../../data/multi-section/data" # Could be adjustable for your own case
        self.datasets = []
        self.current_data = None
        self.current_dataset_name = ""

    def find_datasets(self):
        """Look for all CSV files with RSSI data"""
        print(" Looking for datasets...")

        if not os.path.exists(self.data_folder):
            print(f"Can't find data folder: {self.data_folder}")
            return []

        self.datasets = []

        # Look in each subfolder for TagReadings.csv files
        for folder_name in os.listdir(self.data_folder):
            folder_path = os.path.join(self.data_folder, folder_name)

            if os.path.isdir(folder_path):
                csv_file = os.path.join(folder_path, "TagReadings.csv")

                if os.path.exists(csv_file):
                    file_size = os.path.getsize(csv_file) / (1024 * 1024)

                    self.datasets.append({
                        'name': folder_name,
                        'file_path': csv_file,
                        'size_mb': file_size
                    })

        # Sort by size (biggest first)
        self.datasets.sort(key=lambda x: x['size_mb'], reverse=True)

        print(f" Found {len(self.datasets)} datasets:")
        for i, dataset in enumerate(self.datasets):
            print(f"   {i+1}. {dataset['name']} ({dataset['size_mb']:.1f} MB)")

        return self.datasets

    def load_dataset(self, choice):
        """Load a specific dataset into memory"""
        if choice < 1 or choice > len(self.datasets):
            print(" Invalid choice!")
            return False

        dataset = self.datasets[choice - 1]
        print(f" Loading {dataset['name']}...")

        try:
            self.current_data = pd.read_csv(dataset['file_path'])
            self.current_dataset_name = dataset['name']

            print(f"Loaded {len(self.current_data)} readings")
            return True

        except Exception as e:
            print(f"Error loading file: {e}")
            return False

    def analyze_rssi_basic(self):
        """Do basic analysis of RSSI values"""
        if self.current_data is None:
            print("No data loaded!")
            return

        rssi_values = self.current_data['RSSI']

        print("\n" + "="*50)
        print(f"RSSI Analysis for: {self.current_dataset_name}")
        print("="*50)

        print(f"BASIC STATS:")
        print(f"   Total readings: {len(rssi_values):,}")
        print(f"   Strongest signal: {rssi_values.max():.1f} dBm")
        print(f"   Weakest signal: {rssi_values.min():.1f} dBm")
        print(f"   Average signal: {rssi_values.mean():.1f} dBm")
        print(f"   Middle value: {rssi_values.median():.1f} dBm")

        print(f"\n SIGNAL STRENGTH BREAKDOWN:")
        print(f"   25% of signals are stronger than: {rssi_values.quantile(0.75):.1f} dBm")
        print(f"   50% of signals are stronger than: {rssi_values.quantile(0.50):.1f} dBm")
        print(f"   75% of signals are stronger than: {rssi_values.quantile(0.25):.1f} dBm")

        mean = rssi_values.mean()
        median = rssi_values.median()

        if abs(mean - median) < 2:
            balance = "balanced"
        elif mean > median:
            balance = "has more weak signals than strong ones"
        else:
            balance = "has more strong signals than weak ones"

        print(f"\n  DATA SHAPE: Your data {balance}")

        very_strong = len(rssi_values[rssi_values >= -40])
        strong = len(rssi_values[(rssi_values >= -50) & (rssi_values < -40)])
        medium = len(rssi_values[(rssi_values >= -60) & (rssi_values < -50)])
        weak = len(rssi_values[(rssi_values >= -70) & (rssi_values < -60)])
        very_weak = len(rssi_values[rssi_values < -70])

        total = len(rssi_values)

        print(f"\nSIGNAL CATEGORIES:")
        print(f"   Very Strong (≥-40 dBm): {very_strong:,} ({very_strong/total*100:.1f}%)")
        print(f"   Strong (-50 to -40 dBm): {strong:,} ({strong/total*100:.1f}%)")
        print(f"   Medium (-60 to -50 dBm): {medium:,} ({medium/total*100:.1f}%)")
        print(f"   Weak (-70 to -60 dBm): {weak:,} ({weak/total*100:.1f}%)")
        print(f"   Very Weak (<-70 dBm): {very_weak:,} ({very_weak/total*100:.1f}%)")

        return rssi_values

    def recommend_buckets(self, rssi_values):
        """Recommend how to group the data into buckets"""
        print("\n" + "="*50)
        print("BUCKETING RECOMMENDATION")
        print("="*50)

        # Decide number of buckets
        data_size = len(rssi_values)
        if data_size < 1000:
            num_buckets = 3
        elif data_size < 10000:
            num_buckets = 5
        else:
            num_buckets = 7

        mean = rssi_values.mean()
        median = rssi_values.median()

        if abs(mean - median) < 2:
            strategy = "equal_width"
            reason = "Data looks balanced, so equal-width buckets work well"
        else:
            strategy = "equal_frequency"
            reason = "Data is unbalanced, so equal-frequency buckets are better"

        print(f"RECOMMENDATION:")
        print(f"   Strategy: {strategy.replace('_', ' ').title()}")
        print(f"   Number of buckets: {num_buckets}")
        print(f"   Why: {reason}")

        # Calculate bucket boundaries
        bucket_info = []

        if strategy == "equal_width":
            # Divide the range evenly
            min_rssi = rssi_values.min()
            max_rssi = rssi_values.max()
            bucket_size = (max_rssi - min_rssi) / num_buckets

            print(f"\n EQUAL-WIDTH BUCKETS:")

            for i in range(num_buckets):
                bucket_min = min_rssi + i * bucket_size
                bucket_max = min_rssi + (i + 1) * bucket_size

                # Count how many readings fall in this bucket
                if i == num_buckets - 1:  # Last bucket includes the maximum
                    count = len(rssi_values[(rssi_values >= bucket_min) & (rssi_values <= bucket_max)])
                else:
                    count = len(rssi_values[(rssi_values >= bucket_min) & (rssi_values < bucket_max)])

                bucket_info.append({
                    'bucket': i + 1,
                    'min': bucket_min,
                    'max': bucket_max,
                    'count': count
                })

                print(f"   Bucket {i+1}: {bucket_min:.1f} to {bucket_max:.1f} dBm ({count:,} readings)")

        else:  # equal_frequency
            # Each bucket has roughly the same number of readings
            print(f"\n EQUAL-FREQUENCY BUCKETS:")

            # Calculate percentiles for bucket boundaries
            percentiles = []
            for i in range(num_buckets + 1):
                percentiles.append(i / num_buckets)

            boundaries = [rssi_values.quantile(p) for p in percentiles]

            for i in range(num_buckets):
                bucket_min = boundaries[i]
                bucket_max = boundaries[i + 1]

                # Count readings in this bucket
                if i == num_buckets - 1:  # Last bucket includes the maximum
                    count = len(rssi_values[(rssi_values >= bucket_min) & (rssi_values <= bucket_max)])
                else:
                    count = len(rssi_values[(rssi_values >= bucket_min) & (rssi_values < bucket_max)])

                bucket_info.append({
                    'bucket': i + 1,
                    'min': bucket_min,
                    'max': bucket_max,
                    'count': count
                })

                print(f"   Bucket {i+1}: {bucket_min:.1f} to {bucket_max:.1f} dBm ({count:,} readings)")

        print(f"\n SAMPLING 50% FROM EACH BUCKET:")
        total_sampled = 0
        for bucket in bucket_info:
            sample_size = bucket['count'] // 2
            total_sampled += sample_size
            print(f"   Bucket {bucket['bucket']}: {sample_size:,} readings")

        retention_rate = total_sampled / len(rssi_values) * 100
        print(f"   Total sampled: {total_sampled:,} ({retention_rate:.1f}% of original data)")

        return bucket_info, strategy

    def create_visualization(self, rssi_values, bucket_info, strategy):
        """Create a graph showing the RSSI distribution and buckets"""
        print(f"\n Creating visualization...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram of RSSI values
        ax1.hist(rssi_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('RSSI (dBm)')
        ax1.set_ylabel('Count')
        ax1.set_title(f'RSSI Distribution - {self.current_dataset_name}')
        ax1.grid(True, alpha=0.3)

        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']
        for i, bucket in enumerate(bucket_info):
            color = colors[i % len(colors)]
            ax1.axvline(bucket['min'], color=color, linestyle='--', alpha=0.7,
                       label=f"Bucket {bucket['bucket']} start")

        if bucket_info:
            ax1.axvline(bucket_info[-1]['max'], color='red', linestyle='--', alpha=0.7)

        ax1.legend()

        bucket_numbers = [bucket['bucket'] for bucket in bucket_info]
        bucket_counts = [bucket['count'] for bucket in bucket_info]
        bucket_labels = [f"Bucket {bucket['bucket']}\n({bucket['min']:.1f} to {bucket['max']:.1f})"
                        for bucket in bucket_info]

        bars = ax2.bar(bucket_numbers, bucket_counts, color=colors[:len(bucket_info)],
                      alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Bucket Number')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Bucket Distribution ({strategy.replace("_", " ").title()})')
        ax2.set_xticks(bucket_numbers)
        ax2.set_xticklabels([f"Bucket {i}" for i in bucket_numbers])
        ax2.grid(True, alpha=0.3)

        for bar, count in zip(bars, bucket_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        print("Graph created! Check the pop-up window.")

def main():
    """Main program - this is what runs when you start the program"""
    print("Simple RSSI Analyzer!")
    print("=" * 50)

    analyzer = SimpleRSSIAnalyzer()

    datasets = analyzer.find_datasets()

    if not datasets:
        print("No datasets found. Make sure you're in the right directory!")
        return

    print(f"\n Which dataset would you like to analyze?")
    for i, dataset in enumerate(datasets, 1):
        print(f"   {i}. {dataset['name']} ({dataset['size_mb']:.1f} MB)")

    try:
        choice = int(input(f"\nEnter your choice (1-{len(datasets)}): "))

        if analyzer.load_dataset(choice):
            rssi_values = analyzer.analyze_rssi_basic()

            bucket_info, strategy = analyzer.recommend_buckets(rssi_values)

            analyzer.create_visualization(rssi_values, bucket_info, strategy)

            print(f"\n Analysis of {analyzer.current_dataset_name} completed")

    except ValueError:
        print(" Please enter a valid number!")
    except KeyboardInterrupt:
        print("\n See you next time")

if __name__ == "__main__":
    main()
