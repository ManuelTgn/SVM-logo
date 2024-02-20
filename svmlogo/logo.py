"""
"""

from exception_handlers import exception_handler
from utils import OFFSET
from kmer import Kmer

from igraph import Graph
from typing import List

import numpy as np

import os

class Path():
    def __init__(self, start: int, stop: int, nodes: int, debug: bool) -> None:
        self._debug = debug
        if start != nodes[0] or stop != nodes[-1]:
            exception_handler(ValueError, "Invalid path", os.EX_DATAERR, self._debug)
        self._start = start
        self._stop = stop
        self._path = nodes

    def _get_start(self) -> int:
        return self._start
    
    @property
    def start(self) -> int:
        return self._get_start()

    def _get_stop(self) -> int:
        return self._stop
    
    @property
    def stop(self) -> int:
        return self._get_stop()
    

class SVMLogo():
    def __init__(self, kmers: List[Kmer], debug: bool) -> None:
        self._kmers = kmers
        self._debug = debug

    def _initialize_graph_logo(self, pivot: str) -> None:
        self._logo = Graph().as_directed()
        size = len(pivot) + 1
        for vid in range(len(pivot) + 1):  # add stop vertex
            self.add_vertex(vid)
        self._logo.vs["label"] = list(f"{pivot}*")  # add labels to the graph
        self._paths = [Path(0, size, list(range(size)))]  # initialize paths list

    def _initialize_weight_matrix(self, pivot: str) -> None:
        size = len(pivot) + 1  # motif size
        self._weights = np.zeros((size, size))
        for vid in range(1, size):
            self.add_edge(vid - 1, vid)  # add consecutive edges for pivot
            self._weights[vid - 1, vid] += 1

    def add_vertex(self, vid: int) -> None:
        if vid in self._logo.vs.indices:
            exception_handler(ValueError, "Forbidden vertex addition", os.EX_DATAERR, self._debug)
        self._logo.add_vertex(vid)

    def add_edge(self, parent: int, child: int) -> None:
        if parent not in self._logo.vs.indices or child not in self._logo.vs.indices:
            exception_handler(ValueError, "Forbidden edge addition", os.EX_DATAERR, self._debug)
        self._logo.add_edge(parent, child)

    def _construct_alignment_greedy(self) -> None:
        pivot = self._kmers[0].kmer  # recover pivot sequence
        pivot_weight = self._kmers[0].score
        # initialize graph logo
        self._initialize_graph_logo(pivot)
        # initialize weights matrix
        self._initialize_weight_matrix(pivot)
        # store current largest vertex id and stop node
        self._stop_vid = len(pivot) + 1
        self._largest_vid = self._stop_vid  
        
        

    



