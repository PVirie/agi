from .package_install import install
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