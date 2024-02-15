"""
SVM-logo version {version}

Copyright (C) 2024 Manuel Tognon <manu.tognon@gmail.com> <manuel.tognon@univr.it> <mtognon@mgh.harvard.edu>

Usage:
    svmlogo -m <modelfile>

Run 'svmlogo -h/--help' to display the complete help
"""

from svmlogo_argparse import SVMLogoArgumentParser
from exception_handlers import sigint_handler
from svmlogo_version import __version__

from time import time, sleep

import sys
import os


def parseargs_svmlogo() -> SVMLogoArgumentParser:
    parser = SVMLogoArgumentParser(usage=__doc__, add_help=False)
    group = parser.add_argument_group("Options")
    group.add_argument("-h", "--help", action="help", help="Show this message and exit")
    group.add_argument("-v", "--version", action="version", help="Display SVM-logo version", version=__version__)
    group.add_argument("-m", "--model", type=str, metavar="SVM-model-file", dest="svm_model", required=True, help="SVM-based model file")
    return parser


def main():
    try:
        start = time()
        parser = parseargs_svmlogo()
        if not sys.argv[1:]:  # no input args
            parser.error_noargs()  # display help and exit
            sys.exit(os.EX_USAGE)
        args = parser.parse_args(sys.argv[1:])
        sleep(10)
    except KeyboardInterrupt as e:
        sigint_handler()  # gracefully exit when SIGINT is detected
    stop = time()
    sys.stderr.write(f"SVM-logo - Elapsed time: {(stop - start):.2f}s\n")


# ----------> ENTRY POINT <---------- 
if __name__ == "__main__":
    main()