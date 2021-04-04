"""Printing Helper Functions
"""
from terminaltables import AsciiTable


def as_table(data) -> str:
    """Format data as a table string

    Args:
        data : data

    Returns:
        str: table
    """
    if isinstance(data, dict):
        data = list(data.items())
    return AsciiTable(data).table


def compile_vprint_function(verbose):
    """Compile a verbose print function

    Args:
        verbose (bool): is verbose or not

    Returns:
        [msg, *args]->None: a vprint function
    """

    if verbose:
        return lambda msg, *args: print(msg.format(*args))

    return lambda *_: None
