from .package_install import install, install_pyproject_toml
from datetime import datetime


def get_current_time_string():
    """
    Returns the current time in a formatted string.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_datetime_from_string(date_string):
    """
    Converts a date string in the format 'YYYY-MM-DD HH:MM:SS' to a datetime object.
    """
    return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")


# xterm-256 color index per 0.1 step (truecolor is often stripped by cloud log viewers).
# index 0 -> -1.0, index 10 -> 0.0, index 20 -> 1.0.
_FLOAT_PALETTE = [
    196,  # -1.0 red
    202,  # -0.9
    208,  # -0.8
    214,  # -0.7
    220,  # -0.6
    226,  # -0.5 yellow
    184,  # -0.4
    142,  # -0.3
    100,  # -0.2
    58,   # -0.1
    16,   #  0.0 black
    17,   #  0.1
    18,   #  0.2
    19,   #  0.3
    20,   #  0.4
    21,   #  0.5 blue
    26,   #  0.6
    31,   #  0.7
    36,   #  0.8
    41,   #  0.9
    46,   #  1.0 green
]


def float_to_color_index(f):
    f = max(-1.0, min(1.0, float(f)))
    return _FLOAT_PALETTE[round(f * 10) + 10]


def format_float(f):
    return f"\033[38;5;{float_to_color_index(f)}m{f: .1f}\033[0m"


def scatter(data_list, indices, ops='list'):
    """
    Scatter data from a list of data into a new list based on provided indices and an operation.

    Args:
        data_list (list of *): A list of data items to be scattered.
        indices (list of *): A list of indices corresponding to each data item in data_list.
        ops (str): The operation to apply when scattering data with the same index. Supported operations are 'sum', 'mean', 'max', and 'min'.
    """

    # first correct data for the indices
    accumulator = {}
    for data, key in zip(data_list, indices):
        if key not in accumulator:
            accumulator[key] = []
        accumulator[key].append(data)

    # then apply the operation
    result = {}
    for key, data_list in accumulator.items():
        if ops == 'sum':
            result[key] = sum(data_list)
        elif ops == 'mean':
            result[key] = sum(data_list) / len(data_list)
        elif ops == 'max':
            result[key] = max(data_list)
        elif ops == 'min':
            result[key] = min(data_list)
        else:
            result[key] = data_list
    return result