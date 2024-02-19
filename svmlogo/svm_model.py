"""
"""

from exception_handlers import exception_handler
from support_vector import SupportVector

from itertools import takewhile
from typing import Union, Tuple, List, Set

import os

class SupportVectorModel():
    def __init__(self, model_fname: str, debug: bool) -> None:
        """
        Initializes an instance of SVMModel.

        Args:
            model_fname (str): The file name of the SVM model.
            debug (bool): Flag indicating whether to enable debug mode.

        Raises:
            FileNotFoundError: If the model file is not found.

        Returns:
            None
        """

        self._debug = debug
        if not os.path.isfile(model_fname):
            exception_handler(FileNotFoundError, f"{model_fname} not found", os.EX_OSFILE, self._debug)
        self._model_fname = model_fname
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
            exception_handler(ValueError, "The input model is not a 2-class SV. Unable to build SVM-log", os.EX_DATAERR, self._debug)
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
            exception_handler(ValueError, "Invalid number of support vectors", os.EX_DATAERR, self._debug)
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
            "nr_sv": ("_sv_pos", self._validate_nr_sv)
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
                    field, value = line.strip().split(maxsplit=1)  # ensure key-value split
                    self._set_model_param(field, value)
                # read support vectors
                for _, line in zip(range(self._sv_pos), infile):
                    weight, sequence = line.strip().split()
                    sv = SupportVector(sequence, weight, 0, self._debug)
                    self._support_vectors.append(sv)  
                    assert hasattr(self, "_wlen")
                    self._kmers += sv.kmers_split(self._wlen)
                # remove redundant kmers
                self._kmers = list(set(self._kmers))  
        except OSError as e:
            exception_handler(
                e, f"SVM model loading failed ({self._model_fname})", os.EX_IOERR, self._debug
            )
        
    def _get_support_vectors(self) -> List[SupportVector]:
        return self._support_vectors
    
    @property
    def support_vectors(self) -> List[SupportVector]:
        return self._get_support_vectors()
    
    def _get_kmers(self) -> Set[str]:
        return self._kmers
    
    @property
    def kmers(self) -> Set[str]:
        return self._get_kmers()

