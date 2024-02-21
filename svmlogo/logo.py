"""
"""

from exception_handlers import exception_handler
from utils import OFFSET, ALPHABET
from kmer import Kmer

from igraph import Graph
from typing import List, Tuple

import numpy as np

import os

class Path():
    def __init__(self, start: int, stop: int, nodes: int, sequence: str, debug: bool) -> None:
        self._debug = debug
        if start != nodes[0] or stop != nodes[-1]:
            exception_handler(ValueError, "Invalid path", os.EX_DATAERR, self._debug)
        if len(nodes) - 1 != len(sequence):
            exception_handler(ValueError, "Sequence and vertices length mismatch", os.EX_DATAERR, self._debug)
        self._start = start
        self._stop = stop
        self._path = nodes
        self._sequence = sequence

    def __len__(self) -> int:
        return len(self._path)
    
    def __str__(self) -> str:
        return "-".join(list(map(str, self._path)))
    
    def __getitem__(self, idx: int) -> int:
        if not isinstance(idx, int):
            exception_handler(TypeError, f"Index must be integers, not {type(idx).__name__}", os.EX_DATAERR, self._debug)
        if idx < 0 or idx > len(self):
            exception_handler(IndexError, "Index out of range", os.EX_DATAERR, self._debug)
        return self._path[idx]

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
    
    def _get_sequence(self) -> str:
        return self._sequence
    
    @property
    def sequence(self) -> str:
        return self._get_sequence()
    

class SVMLogo():
    def __init__(self, kmers: List[Kmer], alphabet: int, debug: bool) -> None:
        self._kmers = kmers
        self._debug = debug
        self._alphabet = alphabet
        # construct SVM-logo via greedy procedure
        self._construct_alignment_greedy()

    def _add_vertex_path(self, root: int, leaf: int, vids: List[int], sequence: str) -> None:
        p = Path(root, leaf, vids, sequence, self._debug)
        if not hasattr(self, "_paths"):  # initialize _paths
            self._paths = [p]
        else:
            self._paths.append(p)
        

    def _initialize_graph_logo(self, pivot: str) -> None:
        self._logo = Graph().as_directed()
        size = len(pivot) + 1
        for vid in range(size):  # add stop vertex
            self._add_vertex(vid)
        self._logo.vs["label"] = list(f"{pivot}*")  # add labels to the graph
        nodes = list(range(size))  # pivot sequence path
        self._add_vertex_path(0, size - 1, nodes, f"{pivot}")  # initialize logo paths 

    def _initialize_weight_matrix(self, pivot: str) -> None:
        size = len(pivot) + 1  # motif size
        self._weights = np.zeros((size, size))
        for vid in range(1, size):
            self._add_edge(vid - 1, vid)  # add consecutive edges for pivot
            self._weights[vid - 1, vid] += 1

    def _add_vertex(self, vid: int) -> None:
        if vid in self._logo.vs.indices:
            exception_handler(ValueError, "Forbidden vertex addition", os.EX_DATAERR, self._debug)
        self._logo.add_vertex(vid)

    def _add_edge(self, parent: int, child: int) -> None:
        if parent not in self._logo.vs.indices or child not in self._logo.vs.indices:
            exception_handler(ValueError, "Forbidden edge addition", os.EX_DATAERR, self._debug)
        self._logo.add_edge(parent, child)

    def _match_kmer(self, kmer: Kmer) -> Tuple[str, Path]:
        if not hasattr(self, "_paths"):
            exception_handler(ValueError, "Logo sequences undefined", os.EX_DATAERR, self._debug)
        # align kmer to kmers in the logo
        matches, seqmatch, pathmatch = -1, None, None
        for p in self._paths:
            if len(p) - 1 != len(kmer):  # skip shorter or longer sequences
                continue
            m, s = self._align_kmer(p.sequence, kmer.kmer)
            if m > matches:
                matches, seqmatch, pathmatch = m, s, p
        assert matches >= 0 and seqmatch is not None and pathmatch is not None
        print(kmer.kmer, seqmatch, matches, pathmatch)
        return seqmatch, pathmatch

    def _align_kmer(self, seq1: str, seq2: str) -> Tuple[int, str]:
        matches_counter_right = [self._count_matches(seq1, seq2, start) for start in OFFSET]
        matches_counter_left = [self._count_matches(seq2, seq1, start) for start in OFFSET] 
        # recover best right and left alignments 
        right_align = max(enumerate(matches_counter_right), key=lambda x: sum(x[1]))
        left_align = max(enumerate(matches_counter_left), key=lambda x: sum(x[1]))
        # keep best alignment
        match_nr_right = sum(right_align[1])
        match_nr_left = sum(left_align[1])
        if match_nr_right >= match_nr_left:  # best alignment with offset on the right
            gaps = "".join(["-" for _ in range(right_align[0])])
            seqmatch = "".join([nt if right_align[1][i] else "*" for i, nt in enumerate(seq1[right_align[0]:])])
            return match_nr_right, f"{seqmatch}{gaps}"
        gaps = "".join(["-" for _ in range(left_align[0])])  # best alignment with offset on left
        seqmatch = "".join([nt if left_align[1][i] else "*" for i, nt in enumerate(seq2[left_align[0]:])])
        return match_nr_left, f"{gaps}{seqmatch}"


    def _count_matches(self, query: str, reference: str, start: int) -> int:
        return [nt == reference[i] for i, nt in enumerate(query[start:])]  
    
    def _extend_weights(self, size: int) -> None:
        if size < 1:
            exception_handler(ValueError, "Invalid weight matrix update", os.EX_DATAERR, self._debug)
        m, n = self._weights.shape
        self._weights = np.pad(self._weights, ((0, size), (0, size)))
        m2, n2 = self._weights.shape
        assert m + size == m2 and n + size == n2

    def _parents(self, vid: int) -> List[int]:
        parents = [e.tuple[0] for e in self._logo.es if e.tuple[1] == vid]
        if not parents:
            exception_handler(ValueError, f"Vertex {vid} is not available", os.EX_DATAERR, self._debug)
        return parents
    
    def _update_labels(self, bvertices: int, bstart_vid: int, kmer: Kmer) -> None:
        labels = self.labels
        for i in range(bvertices):
            labels[self._largest_vid - bvertices + 1 + i] = kmer[bstart_vid + i]
        self._logo.vs["label"] = labels


    def _modify_logo(self, kmer: Kmer, seqmatch: str, path: Path) -> None:
        size = len(seqmatch)
        path_current = [None] * (size + 1)  # track the inserted k-mer's path (include stop node) 
        i = 0
        while i < size:
            if seqmatch[i] == "*":  # mismatch, open a bubble
                bstart_pos = i  # bubble start position
                bvertices = 0  # vertices in the current bubble
                while seqmatch[i] == "*" and i < size:  # add vertices to bubble
                    bvertices += 1
                    bvid = self._largest_vid + bvertices  # bubble vertex id
                    self._add_vertex(bvid)
                    path_current[i] = bvid
                    i += 1
                self._largest_vid = bvid  # update largest vertex id
                self._extend_weights(bvertices)  # update weights matrix
                if bstart_pos > 0:  # link parent vertex with first bubble vertex
                    parents = self._parents(path[bstart_pos])  # recover vertex parents
                    bstart_vid = self._largest_vid - bvertices + 1  # bubble starting vertex id
                    for vid in parents:
                        self._add_edge(vid, bstart_vid)
                        self._weights[vid, bstart_vid] += 1  # update logo and weights
                if bvertices > 1:  # link bubble vertices
                    for offset in range(bvertices - 1):
                        bvid_parent = self._largest_vid - bvertices + 1 + offset
                        bvid_child = self._largest_vid - bvertices + 2 + offset
                        self._add_edge(bvid_parent, bvid_child)
                        self._weights[bvid_parent, bvid_child] += 1
                # close the bubble
                if i == size:  # link current path to stop node
                    self._add_edge(self._largest_vid, self._stop_vid)
                    self._weights[self._largest_vid, self._stop_vid] += 1
                    path_current[i] = self._stop_vid
                else:
                    self._add_edge(self._largest_vid, path[i])
                    self._weights[self._largest_vid, path[i]] += 1
                # set labels on bubble vertices
                self._update_labels(bvertices, bstart_pos, kmer)
            else:  # continue along the path
                path_current[i] = path[i]
                i += 1
                if i == size:  # reached the end
                    self._weights[path[i - 1], self._stop_vid] += 1
                    path_current[-1] = self._stop_vid
        self._add_vertex_path(path_current[0], path_current[-1], path_current, kmer.kmer)
        



    def _insert_kmer(self, kmer: Kmer, seqmatch: str, path: Path) -> None:
        offset = seqmatch.count("-")
        if seqmatch[0] in ALPHABET[self._alphabet]:  # no offset
            assert offset == 0
            self._modify_logo(kmer, seqmatch, path)
        for p in self._paths:
            print(p)


        
            


    def _construct_alignment_greedy(self) -> None:
        pivot = self._kmers[0].kmer  # recover pivot sequence
        pivot_weight = self._kmers[0].score
        # initialize graph logo
        self._initialize_graph_logo(pivot)
        # initialize weights matrix
        self._initialize_weight_matrix(pivot)
        # store current largest vertex id and stop node
        self._stop_vid = len(pivot)
        self._largest_vid = self._stop_vid  
        # align kmers to current logo
        for kmer in self._kmers[1:]:
            seqmatch, path = self._match_kmer(kmer)
            self._insert_kmer(kmer, seqmatch, path)

    def _vertex_sequence(self, vids: List[int]) -> str:
        return "".join([self.labels[vid] for vid in vids[:-1]])  # skip stop node (*) 
        
    def _get_labels(self) -> List[str]:
        return self._logo.vs["label"]
    
    @property
    def labels(self) -> List[str]:
        return self._get_labels()
        

    



