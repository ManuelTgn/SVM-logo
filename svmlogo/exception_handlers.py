"""
"""

from colorama import init, Fore
from typing import NoReturn

import sys
import os


def sigint_handler() -> None:
    """
    Handles the SIGINT signal by printing a message and exiting the SVM-logo program.

    Raises:
        None

    Returns:
        None
    """

    sys.stderr.write("\nCaught SIGINT. Exit SVM-logo")
    sys.exit(os.EX_OSERR)


def exception_handler(
    exception_type: Exception, exception: str, code: int, debug: bool
) -> NoReturn:
    """
    Handles exceptions by initializing, displaying error messages, and exiting the program.

    Args:
        exception_type (Exception): The type of the exception.
        exception (str): The error message.
        code (int): The exit code.
        debug (bool): Flag indicating whether to display the full error stack.

    Returns:
        NoReturn

    Raises:
        exception_type: If debug is True, the exception is raised with the error message.

    Examples:
        None
    """

    init()
    if debug:  # display full error stack
        raise exception_type(f"\n\n{exception}")
    # gracefully trigger runtime error and exit
    sys.stderr.write(f"{Fore.RED}\n\nERROR: {exception}\n{Fore.RESET}")
    sys.exit(code)
