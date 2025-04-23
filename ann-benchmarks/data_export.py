import argparse
import os
import datetime
import csv

from ann_benchmarks.datasets import DATASETS, get_dataset
from ann_benchmarks.plotting.utils import compute_metrics_all_runs
from ann_benchmarks.results import load_all_results

# Function to get the current date in yymmdd format
def get_date_str():
    return datetime.datetime.now().strftime("%y%m%d")

# Ensure the reports directory exists
def ensure_reports_dir():
    if not os.path.exists('reports'):
        os.makedirs('reports')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Path to the output file", required=False)
    parser.add_argument("--recompute", action="store_true", help="Recompute metrics")
    args = parser.parse_args()

    # Ensure the reports directory exists
    ensure_reports_dir()

    # Use default file name if no output file is provided
    if not args.output:
        args.output = f"reports/data-{get_date_str()}.csv"
    else:
        # Add date suffix before the extension if output file name is provided
        name, ext = os.path.splitext(args.output)
        # Save the file in the reports directory if no path is given
        if not os.path.dirname(args.output):
            args.output = os.path.join("reports", f"{name}-{get_date_str()}{ext}")
        else:
            args.output = f"{name}-{get_date_str()}{ext}"

    datasets = DATASETS.keys()
    dfs = []
    for dataset_name in datasets:
        print("Looking at dataset", dataset_name)
        if len(list(load_all_results(dataset_name, batch_mode=True))) > 0:
            results = load_all_results(dataset_name, batch_mode=True)
            dataset, _ = get_dataset(dataset_name)
            results = compute_metrics_all_runs(dataset, results, args.recompute)
            for res in results:
                res["dataset"] = dataset_name
                dfs.append(res)
    if len(dfs) > 0:
        with open(args.output, "w", newline="") as csvfile:
            names = list(dfs[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=names)
            writer.writeheader()
            for res in dfs:
                writer.writerow(res)

