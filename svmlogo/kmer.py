"""
"""


class Kmer():
    def __init__(self, kmer: str, score: float) -> None:
        self._kmer = kmer
        self._score = score

    def __str__(self) -> str:
        return f"{self._kmer}\t{self._score}"
    
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