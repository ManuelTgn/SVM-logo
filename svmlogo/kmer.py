"""
"""

from exception_handlers import exception_handler

import os

class Kmer():
    def __init__(self, kmer: str, score: float) -> None:
        self._kmer = kmer
        self._score = score

    def __str__(self) -> str:
        return f"{self._kmer}\t{self._score}"
    
    def __len__(self) -> int:
        return len(self._kmer)
    
    def __getitem__(self, idx: int) -> str:
        if not isinstance(idx, int):
            exception_handler(TypeError, f"Index must be integers, not {type(idx).__name__}", os.EX_DATAERR, True)
        if idx < 0 or idx > len(self):
            exception_handler(IndexError, "Index out of range", os.EX_DATAERR, True)
        return self._kmer[idx] 
    
    def _get_kmer(self) -> str:
        return self._kmer
    
    @property
    def kmer(self) -> str:
        return self._get_kmer()
    
    def _get_score(self) -> float:
        return self._score
    
    @property
    def score(self) -> float:
        return self._get_score()