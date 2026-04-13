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