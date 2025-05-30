import matplotlib as mpl

mpl.use("Agg")  # noqa
import collections
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils import (compute_metrics, create_linestyles,
                                           create_pointset, get_plot_label)
from ann_benchmarks.results import get_unique_algorithms, load_all_results


def create_plot(all_data, raw, x_scale, y_scale, xn, yn, fn_out, linestyles, batch):
    xm, ym = (metrics[xn], metrics[yn])
    # Now generate each plot
    handles = []
    labels = []
    plt.figure(figsize=(12, 9))

    # Sorting by mean y-value helps aligning plots with labels
    def mean_y(algo):
        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        return -np.log(np.array(ys)).mean()

    # Find range for logit x-scale
    min_x, max_x = 1, 0
    for algo in sorted(all_data.keys(), key=mean_y):
        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        min_x = min([min_x] + [x for x in xs if x > 0])
        max_x = max([max_x] + [x for x in xs if x < 1])
        color, faded, linestyle, marker = linestyles[algo]
        #marker = '.'
        #if args.marker:
        #    marker = args.marker

        (handle,) = plt.plot(
            xs, ys, "-", label=algo, color=color, ms=7, mew=3, lw=3, marker=marker
        )
        handles.append(handle)
        if raw:
            (handle2,) = plt.plot(
                axs, ays, "-", label=algo, color=faded, ms=5, mew=2, lw=2, marker=marker
            )
        labels.append(algo)

    ax = plt.gca()
    ax.set_ylabel(ym["description"])
    ax.set_xlabel(xm["description"])
    # Custom scales of the type --x-scale a3
    if x_scale[0] == "a":
        alpha = float(x_scale[1:])

        def fun(x):
            return 1 - (1 - x) ** (1 / alpha)

        def inv_fun(x):
            return 1 - (1 - x) ** alpha

        ax.set_xscale("function", functions=(fun, inv_fun))
        if alpha <= 3:
            ticks = [inv_fun(x) for x in np.arange(0, 1.2, 0.2)]
            plt.xticks(ticks)
        if alpha > 3:
            from matplotlib import ticker

            ax.xaxis.set_major_formatter(ticker.LogitFormatter())
            # plt.xticks(ticker.LogitLocator().tick_values(min_x, max_x))
            plt.xticks([0, 1 / 2, 1 - 1e-1, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, 1])
    # Other x-scales
    else:
        ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_title(get_plot_label(xm, ym))
    plt.gca().get_position()
    # plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 9})
    plt.grid(visible=True, which="major", color="0.65", linestyle="-")
    plt.setp(ax.get_xminorticklabels(), visible=True)

    # Logit scale has to be a subset of (0,1)
    if "lim" in xm and x_scale != "logit":
        x0, x1 = xm["lim"]
        plt.xlim(max(x0, 0), min(x1, 1))
    elif x_scale == "logit":
        plt.xlim(min_x, max_x)
    if "lim" in ym:
        plt.ylim(ym["lim"])

    # Workaround for bug https://github.com/matplotlib/matplotlib/issues/6789
    ax.spines["bottom"]._adjust_location()

    plt.savefig(fn_out, bbox_inches="tight", dpi=144)
    plt.close()

def print_result_info(results):
    # Load just the first result to see its structure
    for properties, f in results:
        print("Properties keys:", list(properties.keys()))
        print("\nProperties values:")
        for key, value in properties.items():
            print(f"  {key}: {value}")

        # Print some info about the file structure if needed
        print("\nFile structure:")
        print("  Keys in file:", list(f.keys()))

        # Only print the first result
        break


def directory_path(s):
    if not os.path.exists(s):
        os.makedirs(s)

    if not os.path.isdir(s):
        raise argparse.ArgumentTypeError("'%s' is not a directory" % s)
    return s + "/"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", metavar="DATASET", default="glove-100-angular")
    parser.add_argument("--count", default=10)
    parser.add_argument(
        "--definitions", metavar="FILE", help="load algorithm definitions from FILE", default="algos.yaml"
    )
    parser.add_argument("--limit", default=-1)
    parser.add_argument("-o", "--output")
    parser.add_argument("--outputdir", help="Select output directory", default="plots/", type=str)
    parser.add_argument("--algo")
    parser.add_argument("--marker")
    parser.add_argument("--dark", help="dark background", action="store_true")
    parser.add_argument(
        "-x", "--x-axis", help="Which metric to use on the X-axis", choices=metrics.keys(), default="k-nn"
    )
    parser.add_argument(
        "-y", "--y-axis", help="Which metric to use on the Y-axis", choices=metrics.keys(), default="qps"
    )
    parser.add_argument(
        "-X", "--x-scale", help="Scale to use when drawing the X-axis. Typically linear, logit or a2", default="linear",
    )
    parser.add_argument(
        "-Y",
        "--y-scale",
        help="Scale to use when drawing the Y-axis",
        choices=["linear", "log", "symlog", "logit"],
        default="linear",
    )
    parser.add_argument(
        "--raw", help="Show raw results (not just Pareto frontier) in faded colours", action="store_true"
    )
    parser.add_argument("--batch", help="Plot runs in batch mode", action="store_true")
    parser.add_argument("--recompute", help="Clears the cache and recomputes the metrics", action="store_true")
    args = parser.parse_args()
    directory_path(args.outputdir)

    if not args.output:
        args.output = args.outputdir + "/%s.png" % (args.dataset + ("-batch" if args.batch else ""))
        print("writing output to %s" % args.output)

    dataset, _ = get_dataset(args.dataset)
    count = int(args.count)
    unique_algorithms = get_unique_algorithms()
    results = load_all_results(args.dataset, count, args.batch)
    linestyles = create_linestyles(sorted(unique_algorithms), args.dark)
    runs = compute_metrics(np.array(dataset["distances"]), results, args.x_axis, args.y_axis, args.recompute)
    if not runs:
        raise Exception("Nothing to plot")

    if args.dark:
        plt.style.use('dark_background')
    
    if args.algo:
        # Step 1: Group all algorithms by their base prefix
        grouped_algos = collections.defaultdict(list)
        algo_to_group = {}

        for algo in unique_algorithms:
            print(algo)
            base = algo.split("-")[0]
            grouped_algos[base].append(algo)
            algo_to_group[algo] = base

            # Step 2: Identify group name of args.algo, if it exists
            args_group = algo_to_group.get(args.algo)

        print(algo_to_group)

        # Step 3: For all other groups, plot args.algo vs the whole group
        for base, group in grouped_algos.items():
            if base == args_group:
                continue  # skip group containing args.algo

            allowed_algorithm_keys = [args.algo] + group
            print(allowed_algorithm_keys)
            filtered_results = {
                algo: results for algo, results in runs.items()
                if algo in allowed_algorithm_keys
            }

            output_file = args.outputdir + f"/{args.algo}-vs-{base}-{args.dataset}.png"
            create_plot(
                filtered_results, args.raw, args.x_scale, args.y_scale,
                args.x_axis, args.y_axis, output_file, linestyles, args.batch
            )

        
    create_plot(
        runs, args.raw, args.x_scale, args.y_scale, args.x_axis, args.y_axis, args.output, linestyles, args.batch
    )
