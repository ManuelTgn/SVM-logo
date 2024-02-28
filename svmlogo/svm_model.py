"""
"""

from exception_handlers import exception_handler
from support_vector import SupportVector
from utils import reverse_complement, ALPHABET
from kmer import Kmer

from itertools import takewhile
from typing import Union, Tuple, List, Set

import subprocess
import os


class SupportVectorModel:
    def __init__(self, model_fname: str, alphabet: int, debug: bool) -> None:
        self._debug = debug
        if not os.path.isfile(model_fname):
            exception_handler(
                FileNotFoundError, f"{model_fname} not found", os.EX_OSFILE, self._debug
            )
        self._model_fname = model_fname
        if alphabet not in ALPHABET:
            exception_handler(
                ValueError, "Unknown alphabet", os.EX_DATAERR, self._debug
            )
        self._alphabet = alphabet
        # read model parameters and support vectors
        self._read_model()

    def __str__(self) -> str:
        """
        Returns a string representation of the SVM model parameters.

        Returns:
            str: A string representation of the SVM model parameters.
        """

        return (
            f"SVM model parameters:\n\t- SVM type: {self._svm}\n\t- Kernel: {self._kernel}"
            f"\n\t- Word length: {self._wlen}\n\t- Informative columns: {self._icols}"
            f"\n\t- Mismatches: {self._mm}\n\t- Numer of classes: {self._class_num}"
            f"\n\t- Total support vectors: {self._svs_total}\n\t- Rho: {self._rho}"
            f"\n\t- Number of support vectors: {self._sv_pos} {self._sv_neg}\n"
        )

    def __len__(self) -> int:
        """
        Returns the number of support vectors in the SVM model.

        Returns:
            int: The number of support vectors in the SVM model.
        """

        return len(self._support_vectors) if hasattr(self, "_support_vectors") else 0

    def _validate_nr_classes(self, value: str) -> int:
        """
        Validates the number of classes in the SVM model.

        Args:
            value (str): The value representing the number of classes.

        Raises:
            ValueError: If the number of classes is not 2.

        Returns:
            int: The validated number of classes.
        """

        value = int(value)
        if value != 2:
            exception_handler(
                ValueError,
                "The input model is not a 2-class SV. Unable to build SVM-log",
                os.EX_DATAERR,
                self._debug,
            )
        return value

    def _validate_nr_sv(self, value: str) -> Tuple[int, int]:
        """
        Validates the number of support vectors for each class in the SVM model.

        Args:
            value (str): The value representing the number of support vectors for each class.

        Raises:
            ValueError: If the number of support vectors is not valid.

        Returns:
            Tuple[int, int]: The validated number of support vectors for each class.
        """

        svs = tuple(map(int, value.split()))
        if len(svs) != 2 or sum(svs) != self._svs_total:
            exception_handler(
                ValueError,
                "Invalid number of support vectors",
                os.EX_DATAERR,
                self._debug,
            )
        return svs

    def _set_model_param(self, field: str, value: str) -> None:
        """
        Sets a model parameter based on the given field and value.

        Args:
            field (str): The field representing the model parameter.
            value (str): The value of the model parameter.

        Returns:
            None
        """

        params_map = {
            "svm_type": ("_svm", str),
            "kernel_type": ("_kernel", str),
            "L": ("_wlen", int),
            "k": ("_icols", int),
            "d": ("_mm", int),
            "nr_class": ("_class_num", self._validate_nr_classes),
            "total_sv": ("_svs_total", int),
            "rho": ("_rho", float),
            "nr_sv": ("_sv_pos", self._validate_nr_sv),
        }
        attr, func = params_map.get(field, (None, None))
        if attr:
            setattr(self, attr, func(value))
            if field == "nr_sv":
                self._sv_neg = self._sv_pos[1]
                self._sv_pos = self._sv_pos[0]

    def _read_model(self) -> None:
        """
        Reads the SVM model from the specified file.

        Raises:
            OSError: If there is an error while reading the model file.

        Returns:
            None
        """

        try:
            self._support_vectors, self._kmers = [], []
            with open(self._model_fname, mode="r") as infile:
                # read svm model header
                header_lines = takewhile(lambda line: "SV" not in line, infile)
                for line in header_lines:
                    field, value = line.strip().split(
                        maxsplit=1
                    )  # ensure key-value split
                    self._set_model_param(field, value)
                # read support vectors
                for _, line in zip(range(self._sv_pos), infile):
                    weight, sequence = line.strip().split()
                    sv = SupportVector(sequence, weight, self._alphabet, self._debug)
                    self._support_vectors.append(sv)
        except OSError as e:
            exception_handler(
                e,
                f"SVM model loading failed ({self._model_fname})",
                os.EX_IOERR,
                self._debug,
            )

    def compute_informative_kmers(self) -> None:
        if not hasattr(self, "_support_vectors"):
            exception_handler(
                ValueError,
                "Support vectors not available, cannot recover informative segments",
                os.EX_DATAERR,
                self._debug,
            )
        kmers_dict = {}
        for sv in self._support_vectors:
            for kmer in sv.kmers_split(self._wlen):
                try:
                    kmers_dict[kmer].append(sv.sequence)
                except KeyError:
                    try:
                        kmers_dict[reverse_complement(kmer, self._alphabet)]
                    except KeyError:
                        kmers_dict[kmer] = [sv.sequence]
        with open(".kmers", mode="w") as outfile:
            for kmer in kmers_dict:
                outfile.write(f">{kmer}\n{kmer}\n")
        subprocess.run(
            ["gkmpredict", "-T 8", ".kmers", self._model_fname, ".scores"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            check=True,
        )
        kmers_scores = {}
        with open(".scores", mode="r") as infile:
            for line in infile:
                kmer, score = line.strip().split()
                kmers_scores[kmer] = float(score)
        informative_kmers = {sv.sequence: ([], []) for sv in self._support_vectors}
        for kmer in kmers_dict:
            for seq in kmers_dict[kmer]:
                informative_kmers[seq][0].append(kmer)
                informative_kmers[seq][1].append(kmers_scores[kmer])
        self._informative_kmers = [
            Kmer(
                informative_kmers[seq][0][
                    informative_kmers[seq][1].index(max(informative_kmers[seq][1]))
                ],
                max(informative_kmers[seq][1]),
            )
            for seq in informative_kmers
        ]
        assert len(self._informative_kmers) == self._sv_pos
        subprocess.run(["rm", ".kmers", ".scores"])
        # sort informative k-mers by weight
        self._informative_kmers = sorted(
            self._informative_kmers, key=lambda x: x.score, reverse=True
        )

    @property
    def support_vectors(self) -> List[SupportVector]:
        return self._support_vectors

    @property
    def kmers(self) -> Set[str]:
        return self._kmers

    @property
    def informative_kmers(self) -> List[Kmer]:
        return self._informative_kmers

    @property
    def wlen(self) -> int:
        assert hasattr(self, "_wlen")
        return self._wlen
