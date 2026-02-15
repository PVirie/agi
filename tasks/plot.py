import os
import sys
import logging
from datetime import datetime
import argparse
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib import lines, markers
from cycler import cycler
import tkinter as tk


def open_file_dialog():
    """Opens a file dialog to select a single file."""
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Statistic file",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )

    if file_path:
        return file_path
    else:
        logging.warning("No file selected.")
        return None


def open_files_dialog():
    """Opens a file dialog to select multiple files."""
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(
        title="Select Rollout files",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )

    if file_paths:
        for file_path in file_paths:
            yield file_path


def save_file_dialog(default_path = None):
    """Opens a file dialog to save a file."""
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.asksaveasfilename(
        title="Save Plot",
        defaultextension=".pgf",
        filetypes=[("PGF Files", "*.pgf"), ("All Files", "*.*")],
        initialfile=default_path if default_path else "plot.pgf"
    )
    if file_path:
        return file_path
    else:
        logging.warning("No file selected for saving.")
        return None


def parse_statistic_file(file_path, aggregate_steps=1000):
    """
    The input is a csv file with columns: namespace/game_id/metric_name, ...
    Return dict of game_id: where is game_id is also a dict of metric: that contains a list of mean and variance for each step. 
    For example:
    {
        "game_1": {
            "return": {
                "mean": [0, 1, 2, ...],
                "var": [0, 0.5, 1.0, ...],
            },
            "episode_length": {
                "mean": [0, 0.5, 1.5, ...],
                "var": [0, 0.25, 0.75, ...],
            },
            ...
        },

    aggregate_steps: we will aggregate the data into steps of this size. 
    For example, if aggregate_steps=1000, then we will compute the mean and variance for steps 0-999, 1000-1999, etc. 
    This is useful for smoothing the curves when plotting.
    """

    game_data = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        # parse header to get metric names and game ids
        info = []
        for col in header:
            parts = col.split('/')
            if len(parts) != 3:
                logging.warning(f"Unexpected column name format: {col}")
                continue
            metric_name = parts[-1]
            game_id = "/".join(parts[:-1])
            info.append((metric_name, game_id))

        stats = {}
        for metric_name, game_id in info:
            if game_id not in game_data:
                game_data[game_id] = {}
                stats[game_id] = {}
            if metric_name not in game_data[game_id]:
                game_data[game_id][metric_name] = {
                    "mean": [],
                    "var": []
                }
                stats[game_id][metric_name] = {
                    "sum_x": 0,
                    "sum_x2": 0,
                    "count": 0
                }

        for j, row in enumerate(reader):
            # first append 0 for each row
            if j % aggregate_steps == 0:
                if j > 0:
                    # compute mean and variance for the previous batch
                    for metric_name, game_id in info:
                        stat = stats[game_id][metric_name]
                        count = stat["count"]
                        if count > 0:
                            mean = stat["sum_x"] / count
                            var = (stat["sum_x2"] / count) - (mean * mean)
                            game_data[game_id][metric_name]["mean"].append(mean)
                            game_data[game_id][metric_name]["var"].append(var)
                        else:
                            logging.warning(f"No valid data for metric '{metric_name}' and game '{game_id}' in batch ending at row {j}")
                            game_data[game_id][metric_name]["mean"].append(0)
                            game_data[game_id][metric_name]["var"].append(0)

                for metric_name, game_id in info:
                    stat = stats[game_id][metric_name]
                    stat["sum_x"] = 0
                    stat["sum_x2"] = 0
                    stat["count"] = 0

            for i, value in enumerate(row):
                if i >= len(info):
                    logging.warning(f"More columns than expected in row: {row}")
                    break
                metric_name, game_id = info[i]
                try:
                    x = float(value)
                except ValueError:
                    logging.warning(f"Non-numeric value '{value}' for metric '{metric_name}' and game '{game_id}' in row: {row}")
                    continue

                stat = stats[game_id][metric_name]
                stat["sum_x"] += x
                stat["sum_x2"] += x * x
                stat["count"] += 1

        # finalize the last batch if it is not complete
        for metric_name, game_id in info:
            stat = stats[game_id][metric_name]
            count = stat["count"]
            if count > 0:
                mean = stat["sum_x"] / count
                var = (stat["sum_x2"] / count) - (mean * mean)
                game_data[game_id][metric_name]["mean"].append(mean)
                game_data[game_id][metric_name]["var"].append(var)
            else:
                logging.warning(f"No valid data for metric '{metric_name}' and game '{game_id}' in the last batch")
                game_data[game_id][metric_name]["mean"].append(0)
                game_data[game_id][metric_name]["var"].append(0)

    return game_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Plot Task")

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # print summary of arguments that are not default
    logging.info("With arguments:")
    for arg in vars(args):
        value = getattr(args, arg)
        default = parser.get_default(arg)
        if value != default:
            logging.info(f"  --{arg}: {value} (default: {default})")

    aggregate_steps = 1000
    data = parse_statistic_file(open_file_dialog(), aggregate_steps=aggregate_steps)

    # Plotting the data with error bar
    style_cycler = cycler(
        color=plt.cm.tab10.colors[: 4],  # use first 4 colors from tab10
        linestyle=['-', '-.', ':', '--'],
    )
    fig, axs = plt.subplots(3, 1, figsize=(9, 16))
    for ax in axs:
        ax.set_prop_cycle(style_cycler)
    for k, (game_id, game_data) in enumerate(data.items()):
        for j, (metric_name, stats) in enumerate(game_data.items()):
            ax = axs[j]
            X = [i * aggregate_steps for i in range(len(stats["mean"]))]
            Y = stats["mean"]
            STD = [var ** 0.5 for var in stats["var"]]
            # normalize Y and STD by their max value for better visualization
            max_Y = max(abs(y) for y in Y) if Y else 1
            Y = [y / max_Y for y in Y]
            STD = [s / max_Y for s in STD]
            label = f"{game_id}"
            # ax.errorbar(X, Y, yerr=STD, markersize=3, capsize=3, label=label, elinewidth=1, markeredgewidth=1, linewidth=2)
            ax.plot(X, Y, markersize=3, label=label, linewidth=2)
            ax.fill_between(X, [Y[i] - STD[i] for i in range(len(Y))], [Y[i] + STD[i] for i in range(len(Y))], alpha=0.1)
            ax.set_xlabel('step')
            ax.set_ylabel(metric_name)
            # ax.set_title('Rollout Scores')
            ax.legend()
    fig.tight_layout()
    plt.show()


    # plot_name = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pgf"
    # save_path = save_file_dialog(default_path=plot_name)
    # if save_path is not None:
    #     logging.info(f"Saving plot to {save_path}")
    #     fig.savefig(save_path, format="pgf", bbox_inches="tight")
    #     # now clear all the header sections from the file starts with %%
    #     with open(save_path, 'r') as f:
    #         lines = f.readlines()
    #     with open(save_path, 'w') as f:
    #         for i, line in enumerate(lines):
    #             if line.startswith('%') and i >= 8:
    #                 continue
    #             f.write(line)

    plt.close(fig)