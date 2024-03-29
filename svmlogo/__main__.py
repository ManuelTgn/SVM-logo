"""
SVM-logo version {version}

Copyright (C) 2024 Manuel Tognon <manu.tognon@gmail.com> <manuel.tognon@univr.it> <mtognon@mgh.harvard.edu>

Usage:
    svmlogo -m <modelfile>

Run 'svmlogo -h/--help' to display the complete help
"""

from svmlogo_argparse import SVMLogoArgumentParser
from svm_model import SupportVectorModel
from exception_handlers import sigint_handler
from svmlogo_version import __version__
from logo import SVMLogo
from kmer import Kmer

from itertools import product
from argparse import Namespace
from time import time, sleep

import pandas as pd

import sys
import os


def parseargs_svmlogo() -> SVMLogoArgumentParser:
    """
    Returns an instance of SVMLogoArgumentParser configured for SVM-logo command-line arguments.

    Returns:
        SVMLogoArgumentParser: An instance of SVMLogoArgumentParser configured for SVM-logo command-line arguments.
    """

    parser = SVMLogoArgumentParser(usage=__doc__, add_help=False)
    group = parser.add_argument_group("Options")
    group.add_argument("-h", "--help", action="help", help="Show this message and exit")
    group.add_argument(
        "-v",
        "--version",
        action="version",
        help="Display SVM-logo version",
        version=__version__,
    )
    group.add_argument(
        "-m",
        "--model",
        type=str,
        metavar="SVM-model-file",
        dest="svm_model",
        required=True,
        help="SVM-based model file",
    )
    group.add_argument(
        "-a",
        "--alphabet",
        type=int,
        metavar="ALPHABET",
        nargs="?",
        default=0,
        help="Support vectors alphabet. Available values: DNA = 0, RNA = 1, AMINOACID = 2 [default DNA]",
    )
    group.add_argument(
        "-o",
        "--outfile",
        type=str,
        metavar="OUTFILE",
        required=True,
        help="Output logo file name",
    )
    group.add_argument(
        "--debug", action="store_true", default=False, help="Trace the full error stack"
    )
    return parser


def svmlogo(args: Namespace) -> None:
    # build support vector model from input file
    # svm = SupportVectorModel(args.svm_model, args.alphabet, args.debug)
    # svm.compute_informative_kmers()
    # print(len(svm._informative_kmers), len(set(svm._informative_kmers)))
    # with open("kmers_scores.txt", mode="w") as outfile:
    #     for kmer in svm.informative_kmers:
    #         outfile.write(f"{kmer.kmer}\t{kmer.score}\n")
    # compute SVM-logo from input model
    # kmers = [Kmer("AGAGATAAGA", 1), Kmer("CAGAGCTATG", 1)] #, Kmer("GATAAGACCC", 1),  Kmer("CGAGATAAGA", 1), Kmer("AGAGATAAAC", 1)]
    # kmers = [Kmer("AGAGATAAGA", 1), Kmer("TAGAGATAAG", 1), Kmer("TCTTATCTCT", 1), Kmer("AGATAAGATT", 1), Kmer("AGATAAGGAA", 1), Kmer("TTCCTTATCT", 1), Kmer("AGAGATAAGG", 1), Kmer("CCTTATCTCT", 1)]
    # kmers = [Kmer("AGAGATAAGA", 1), Kmer("GATTAGACCC", 1), Kmer("AGAGACAAGA", 1), Kmer("AGAGATAAGA", 1), Kmer("TGAGACAAGA", 1), Kmer("CCAGAGATAA", 1), Kmer("AGAGATAAGA", 1), Kmer("AGAGATAAGA", 1), Kmer("AGAGATAAGA", 1), Kmer("AGAGATAAGA", 1)]
    with open("kmers_scores.txt", mode="r") as infile:
        kmers = [
            Kmer(fields[0], float(fields[1]))
            for line in infile
            for fields in [line.strip().split()]
        ]
    logo = SVMLogo(kmers, 12, 0.1, args.alphabet, args.debug)
    logo.display(args.outfile, True)  # display logo


def main():
    try:
        start = time()
        parser = parseargs_svmlogo()
        if not sys.argv[1:]:  # no input args
            parser.error_noargs()  # display help and exit
            sys.exit(os.EX_USAGE)
        # construct logo
        svmlogo(parser.parse_args(sys.argv[1:]))
    except KeyboardInterrupt as e:
        sigint_handler()  # gracefully exit when SIGINT is detected
    stop = time()
    sys.stderr.write(f"SVM-logo - Elapsed time: {(stop - start):.2f}s\n")


# ----------> ENTRY POINT <----------
if __name__ == "__main__":
    main()
