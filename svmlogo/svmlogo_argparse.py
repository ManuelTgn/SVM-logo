"""
This script defines the SVMLogoArgumentParser class, a custom argument parser 
for SVM-logo. It extends the ArgumentParser class from the argparse module.

Classes:
    SVMLogoArgumentParser: A custom argument parser for SVM-logo.

    SVMLogoArgumentParser.SVMLogoHelpFormatter: A custom help formatter for 
                                                    SVMLogo.

Methods:
    SVMLogoArgumentParser.__init__: Initializes the SVMLogoArgumentParser.
    SVMLogoArgumentParser.error: Displays an error message and exits.
    SVMLogoArgumentParser.error_noargs: Prints help message and exits.

Helper Functions:
    capture_stderr: Captures stderr output for testing purposes.

Test Functions:
    test_happy_path: Parametrized test for the happy path.
    test_error_method: Parametrized test for the error method.
    test_error_noargs_method: Parametrized test for the error_noargs method.
    test_custom_help_formatter_add_usage: Parametrized test for the custom help 
                                            formatter add_usage method.
"""

from svmlogo_version import __version__

from argparse import SUPPRESS, ArgumentParser, HelpFormatter
from typing import NoReturn, Optional, Tuple, Dict
from colorama import Fore

import sys
import os


class SVMLogoArgumentParser(ArgumentParser):
    """A custom argument parser for SVM-logo.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        SVMLogoHelpFormatter: A custom help formatter for SVMLogo.

    Methods:
        __init__: Initializes the SVMLogoArgumentParser.
        error: Displays an error message and exits.
        error_noargs: Prints help message and exits.
    """

    class SVMLogoHelpFormatter(HelpFormatter):
        """A custom help formatter for SVMLogo.

        Args:
            usage (str): The usage string.
            actions (str): The actions string.
            groups (str): The groups string.
            prefix (Optional[str]): The prefix string. Defaults to None.

        Methods:
            add_usage: Adds usage information to the help formatter.
        """

        def add_usage(
            self, usage: str, actions: str, groups: str, prefix: Optional[str] = "None"
        ) -> None:
            """Adds usage information to the help formatter.

            Args:
                usage (str): The usage string.
                actions (str): The actions string.
                groups (str): The groups string.
                prefix (Optional[str]): The prefix string. Defaults to None.
            """

            if usage != SUPPRESS:
                args = (usage, actions, groups, "")
                self._add_item(self._format_usage, args)

    def __init__(self, *args: Tuple, **kwargs: Dict) -> None:
        """Initializes the SVMLogoArgumentParser.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        kwargs["formatter_class"] = self.SVMLogoHelpFormatter
        kwargs["usage"] = kwargs["usage"].replace("{version}", __version__)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> NoReturn:
        """Displays an error message and exits.

        Args:
            message (str): The error message.
        """

        errmessage = (
            Fore.RED + "\nERROR: " + f"{message}." + Fore.RESET
        )  # display error message in red
        errmessage += f"\n\nRun svmlogo -h for usage\n\n"
        sys.stderr.write(message)  # write error message to stderr
        sys.exit(os.EX_USAGE)

    def error_noargs(self) -> None:
        """Prints help message and exits."""

        self.print_help()
        sys.exit(os.EX_USAGE)
