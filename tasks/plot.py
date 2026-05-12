import os
import sys
import logging
from datetime import datetime
import argparse
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib import lines, markers
from cycler import cycler
import tkinter as tk

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utilities import episode_recorder as csv


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

    file_paths = []
    more = True
    
    while more:
        files = filedialog.askopenfilenames(
            title="Select Statistic files",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if files:
            file_paths.extend(files)
            more = tk.messagebox.askyesno("Select More Files", "Do you want to select more files?")
        else:
            more = False

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
    

def compute_common_part(file_paths, separator=os.sep, suffix=False):
    parts = [file_path.split(os.sep) for file_path in file_paths]
    if suffix:
        # reverse the parts for common suffix
        parts = [list(reversed(part)) for part in parts]
    common_part = []
    for i in range(min(len(part) for part in parts)):
        current_part = parts[0][i]
        if all(part[i] == current_part for part in parts):
            common_part.append(current_part)
        else:
            break
    if suffix:
        common_part = list(reversed(common_part))
    return separator.join(common_part)


def parse_statistic_file(file_path_generator, aggregate_steps=1000, N=50):
    """
    The input is a csv file with columns: namespace/game_id/metric_name, ...
    Return dict of game_id: where is game_id is also a dict of metric: that contains a list of mean and variance for each step. 
    For example:
    {
        "file_1/game_1": {
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

    N: averaging window size for computing mean and variance. For example, if N=50, then we will compute the mean and variance for the last 50 steps in each batch. 
    """

    # first get all file paths to compute shortest unique path (sup) for better visualization
    file_paths = set()
    for file_index, file_path in enumerate(file_path_generator):
        file_paths.add(file_path)

    # compute common prefix (quantize by /, to prevent cutting off part)
    cp = compute_common_part(file_paths, separator=os.sep, suffix=False)

    # compute common suffix (quantize by /, to prevent cutting off part)
    cs = compute_common_part(file_paths, separator=os.sep, suffix=True)
        
    # subtract common prefix up to the common suffix, to get the unique part of the file path for better visualization
    unique_file_paths = []
    for file_path in file_paths:
        sup = file_path[len(cp):len(file_path)-len(cs)] if len(cs) > 0 else file_path[len(cp):]
        # trim
        if sup.startswith(os.sep):
            sup = sup[len(os.sep):]
        if sup.endswith(os.sep):
            sup = sup[:-len(os.sep)]
        unique_file_paths.append((sup, file_path))


    game_data = {}
    for (sup, file_path) in unique_file_paths:
        logging.info(f"Parsing file: {sup}")
        with open(file_path, 'rb') as f:
            reader = csv.reader(f)
            header = next(reader)

            # parse header to get metric names and game ids
            info = []
            for col in header:
                parts = col.split('/')
                if len(parts) < 2:
                    logging.warning(f"Unexpected column name format: {col}")
                    continue
                metric_name = parts[-1]
                game_id = f"{sup}/" + "/".join(parts[:-1])
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
                    stats[game_id][metric_name] = []


            def add_datapoint():
                # compute mean and variance for the previous batch
                for metric_name, game_id in info:
                    stat = stats[game_id][metric_name]
                    count = len(stat)
                    if count > 0:
                        sum_stat = sum(stat)
                        sum_stat2 = sum(x ** 2 for x in stat)
                        mean = sum_stat / count
                        var = (sum_stat2 / count) - (mean * mean)
                        game_data[game_id][metric_name]["mean"].append(mean)
                        game_data[game_id][metric_name]["var"].append(var)
                    else:
                        logging.warning(f"No valid data for metric '{metric_name}' and game '{game_id}' in batch ending at row {j}")
                        game_data[game_id][metric_name]["mean"].append(0)
                        game_data[game_id][metric_name]["var"].append(0)


            for j, row in enumerate(reader):
                # first append 0 for each row
                if j % aggregate_steps == 0 and j > 0:
                    add_datapoint()

                for i, value in enumerate(row):
                    if i >= len(info):
                        logging.warning(f"More columns than expected in row: {row}")
                        break
                    metric_name, game_id = info[i]
                    try:
                        x = float(value)
                    except ValueError:
                        # logging.warning(f"Non-numeric value '{value}' for metric '{metric_name}' and game '{game_id}' in row: {row}")
                        continue

                    stats[game_id][metric_name].append(x)
                    while len(stats[game_id][metric_name]) > N:
                        stats[game_id][metric_name].pop(0)
            else:
                # finalize the last batch if it is not complete
                add_datapoint()

    return game_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Plot Task")

    parser = argparse.ArgumentParser()
    parser.add_argument("--normalize",              "-n",   action="store_false")
    parser.add_argument("--ma-window-size",         "-m",   type=int, default=10000, help="Moving average window size for smoothing the curves")
    args = parser.parse_args()

    # print summary of arguments that are not default
    logging.info("With arguments:")
    for arg in vars(args):
        value = getattr(args, arg)
        default = parser.get_default(arg)
        if value != default:
            logging.info(f"  --{arg}: {value} (default: {default})")

    aggregate_steps = 1000
    data = parse_statistic_file(open_files_dialog(), aggregate_steps=aggregate_steps)

    # Plotting the data with error bar
    style_cycler = cycler(
        color=plt.cm.tab10.colors,
        # line start start from solid and change frequency of dot and dash to make it more distinguishable
        linestyle=['-', (0, (5, 1)), (0, (3, 1)), (0, (1, 1)), (0, (5, 1, 3, 1)), (0, (3, 1, 1, 1)), (0, (1, 1, 1, 1)), (0, (5, 1, 1, 1, 1, 1)), (0, (3, 1, 3, 1, 1, 1)), (0, (3, 1, 1, 1, 3, 1))],
    )
    fig, axs = plt.subplots(3, 1, figsize=(9, 16))
    for ax in axs:
        ax.set_prop_cycle(style_cycler)
    for k, (game_id, game_data) in enumerate(data.items()):
        for j, (metric_name, stats) in enumerate(game_data.items()):
            ax = axs[j]
            X = [i * aggregate_steps for i in range(len(stats["mean"]))]
            if args.ma_window_size > 1:
                # apply moving average to Y and STD
                Y = []
                STD = []
                for i in range(len(stats["mean"])):
                    start = max(0, i - args.ma_window_size + 1)
                    end = i + 1
                    mean_window = stats["mean"][start:end]
                    var_window = stats["var"][start:end]
                    Y.append(sum(mean_window) / len(mean_window))
                    STD.append((sum(var_window) / len(var_window)) ** 0.5)
            else:
                Y = stats["mean"]
                STD = [var ** 0.5 for var in stats["var"]]
            if args.normalize:
                # normalize Y and STD by their ranges for better visualization
                min_Y = min(Y)
                max_Y = max(Y)
                Y = [(y - min_Y) / (max_Y - min_Y) for y in Y]
                STD = [std / (max_Y - min_Y) for std in STD]
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