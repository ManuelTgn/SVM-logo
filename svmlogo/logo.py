"""
"""

from exception_handlers import exception_handler
from utils import OFFSET, ALPHABET, CENTRAL, RIGHT, LEFT
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
    
    @property
    def start(self) -> int:
        return self._start
    
    @property
    def stop(self) -> int:
        return self._stop
    
    @property
    def sequence(self) -> str:
        return self._sequence
    

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
    
    def _update_labels(self, bvertices: int, bstart_vid: int, kmer: str) -> None:
        labels = self.labels
        for i in range(bvertices):
            labels[self._largest_vid - bvertices + 1 + i] = kmer[bstart_vid + i]
        self._logo.vs["label"] = labels

    def _update_weights(self, path: List[int]) -> None:
        for i, vid in enumerate(path[:-1]):
            self._weights[vid, path[i + 1]] += 1

    def _extend_bubble(self, seqmatch: str, start: int, size: int, path: List[int]) -> Tuple[int, int, List[int], int]:
        pos = start  # start extension
        bsize = 0  # bubble size
        while seqmatch[pos] == "*" and pos < size:
            bsize += 1
            bvid = self._largest_vid + bsize  # bubble vertex id
            self._add_vertex(bvid)  # add vertex to bubble
            path[pos] = bvid  # update k-mer's path
            pos += 1
            if pos == size:  # bubble closes at k-mer's end
                return bsize, bvid, path, pos
        return bsize, bvid, path, pos
    
    def _connect_bubble(self, bsize: int) -> None:
        for offset in range(bsize - 1):  # avoid dead link for last bubble vertex
            bvid_parent = self._largest_vid - bsize + 1 + offset
            bvid_child = self._largest_vid - bsize + 2 + offset
            self._add_edge(bvid_parent, bvid_child)  # link vertices

    def _anchor_bubble(self, path: Path, start: int, bsize: int) -> None:
        parents = self._parents(path[start])  # recover bubble parents
        bstart_vid = self._largest_vid - bsize + 1  # bubble start vertex id
        for vid in parents:  # anchor bubble vertices to parents
            self._add_edge(vid, bstart_vid)

    def _close_bubble(self, pos: int, size: int, path_current: List[int], path: Path) -> None:
        if pos == size:  # link the bubble to stop node
            self._add_edge(self._largest_vid, self._stop_vid)
            path_current[pos] = self._stop_vid  # add stop vertex to current path
        else:  # link bubble to existing vertex
            self._add_edge(self._largest_vid, path[pos])

    def _extend_offset(self, offset: int, path: List[int], pos: int) -> Tuple[int, List[int]]:
        for i in range(offset):
            ovid = self._largest_vid + i + 1
            self._add_vertex(ovid)  # add offset vertex
            path[pos + i] = ovid  # update current path
        return ovid, path
    
    def _anchor_offset(self, ovid: int, vid: int, direction: str) -> None:
        if direction == LEFT:
            self._add_edge(ovid, vid)
        elif direction == RIGHT:
            self._add_edge(vid, ovid)

    def _modify_logo(self, kmer: Kmer, seqmatch: str, path: Path, offset: int, direction: str) -> None:
        size = len(seqmatch)
        path_current = [None] * (size + 1)  # track the inserted k-mer's path (include stop node) 
        path_to_add = False
        start, stop = 0, size
        if direction == LEFT:
            start = offset


        if direction == LEFT:
            self._largest_vid, path_current = self._extend_offset(offset, path_current, 0)
            self._extend_weights(offset)
            self._connect_bubble(offset)
            ovid = self._largest_vid
            start = offset
            size -= offset            
            path_to_add = True
        sequence = kmer.kmer[start:]
        seqmatch = seqmatch[start:]
        for i, c in enumerate(seqmatch):
            pos = i + start
            if c == "*":  # mismatch, open a bubble
                bstart = pos + start  # bubble start position
                path_to_add = True  # add path since it diverges 
                bvertices, self._largest_vid, path_current, i = self._extend_bubble(seqmatch, bstart, size, path_current)
                self._extend_weights(bvertices)  # update weights matrix
                if bstart > 0:  # link parent vertex with first bubble vertex
                    self._anchor_bubble(path, bstart, bvertices)
                if bvertices > 1:  # link bubble vertices
                    self._connect_bubble(bvertices)
                # close the bubble and set labels on bubble vertices
                self._close_bubble(pos, size, path_current, path)
                self._update_labels(bvertices, bstart, sequence)
            else:  # match, continue along the path
                path_current[pos] = path[pos]
                if pos == size - 1:
                    path_current[-1] = self._stop_vid
        self._update_weights(path_current)  # update weights matrix
        if path_to_add:
            self._add_vertex_path(path_current[0], path_current[-1], path_current, kmer.kmer)
        for e in self._logo.es:
            print(e.tuple)

    


    def _insert_kmer(self, kmer: Kmer, seqmatch: str, path: Path) -> None:
        offset = seqmatch.count("-")
        direction = CENTRAL
        if offset > 0:
            direction = LEFT if seqmatch.startswith("-") else RIGHT
        self._modify_logo(kmer, seqmatch, path, offset, direction)



        # elif seqmatch[0] == "-":  # offset on the left
        #     assert offset > 0
        #     self._modify_logo(kmer.kmer[offset:], seqmatch[offset:], path, LEFT, offset)


        
            


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
        print(self._weights)
        for p in self._paths:
            print(p)

    def _vertex_sequence(self, vids: List[int]) -> str:
        return "".join([self.labels[vid] for vid in vids[:-1]])  # skip stop node (*) 
    
    @property
    def labels(self) -> List[str]:
        return self._logo.vs["label"]
        

    



