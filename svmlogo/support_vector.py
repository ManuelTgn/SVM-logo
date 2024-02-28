"""
"""

from exception_handlers import exception_handler
from utils import DNA, RNA, ALPHABET  # TODO: handle aminoacid alphabet

from typing import List, Set

import os


class SupportVector:
    def __init__(
        self, sequence: str, weight: float, alphabet: int, debug: bool
    ) -> None:
        self._debug = debug
        self._read_sequence(
            sequence, alphabet
        )  # read and evaluate support vector sequence
        self._weight = weight

    def __str__(self) -> str:
        return self._sequence

    def __len__(self) -> int:
        return len(self._sequence)

    def _read_sequence(self, sequence: str, alphabet: int) -> None:
        sequence = sequence.upper()  # avoid case mismatch
        # check input sequence consistency according to input alphabet
        if alphabet == DNA:
            if any(nt not in ALPHABET[DNA] for nt in sequence):
                exception_handler(
                    ValueError,
                    "The input sequence is not a valid DNA string",
                    os.EX_DATAERR,
                    self._debug,
                )
        elif alphabet == RNA:
            if any(nt not in ALPHABET[RNA] for nt in sequence):
                exception_handler(
                    ValueError,
                    "The input sequence is not a valid RNA string",
                    os.EX_DATAERR,
                    self._debug,
                )
        self._sequence = sequence

    def kmers_split(self, k: int) -> Set[str]:
        """
        Splits the support vector sequence into k-mers of length k.

        Args:
            k (int): The length of the k-mers.

        Raises:
            ValueError: If the support vector sequence is undefined.

        Returns:
            List[str]: A list of unique k-mers from the support vector sequence.
        """

        if hasattr(self, "_sequence"):
            return list({self._sequence[i : i + k] for i in range(len(self) - k + 1)})
        exception_handler(
            ValueError, "Support vector sequence undefined", os.EX_DATAERR, self._debug
        )

    def _get_sequence(self) -> str:
        return self._sequence

    @property
    def sequence(self) -> str:
        return self._get_sequence()

    def _get_weight(self) -> float:
        return self._weight

    @property
    def weight(self) -> float:
        return self._get_weight()
