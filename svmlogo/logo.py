"""
"""

from exception_handlers import exception_handler
from utils import (
    OFFSET,
    ALPHABET,
    ALPHABETMAP,
    CENTRAL,
    RIGHT,
    LEFT,
    PALETTE,
    reverse_complement,
)
from kmer import Kmer
from svmlogo_version import __version__

from tqdm import tqdm
from io import TextIOWrapper
from igraph import Graph
from typing import List, Tuple, Union, Dict

import numpy as np

import itertools
import datetime
import os


class Path:
    """Represents a path with start and stop indices, nodes, sequence, and debug flag.

    Args:
        start: The start index of the path.
        stop: The stop index of the path.
        nodes: A list of integers representing the nodes in the path.
        sequence: A string representing the sequence associated with the path.
        debug: A boolean flag indicating whether debug mode is enabled.

    Raises:
        ValueError: If start or stop do not match the first or last node in the path respectively,
            or if the length of nodes minus 1 does not match the length of the sequence.

    Attributes:
        start: The start index of the path.
        stop: The stop index of the path.
        path: A list of integers representing the nodes in the path.
        sequence: A string representing the sequence associated with the path.
    """

    def __init__(
        self, start: int, stop: int, nodes: List[int], sequence: str, debug: bool
    ) -> None:
        """Initializes a Path object.

        Args:
            start: The start index of the path.
            stop: The stop index of the path.
            nodes: A list of integers representing the nodes in the path.
            sequence: A string representing the sequence associated with the path.
            debug: A boolean flag indicating whether debug mode is enabled.

        Raises:
            ValueError: If start or stop do not match the first or last node in the path respectively,
                or if the length of nodes minus 1 does not match the length of the sequence.
        """

        self._debug = debug
        if start != nodes[0] or stop != nodes[-1]:
            exception_handler(ValueError, "Invalid path", os.EX_DATAERR, self._debug)
        if len(nodes) - 1 != len(sequence):
            exception_handler(
                ValueError,
                "Sequence and vertices length mismatch",
                os.EX_DATAERR,
                self._debug,
            )
        self._start = start
        self._stop = stop
        self._path = nodes
        self._sequence = sequence

    def __len__(self) -> int:
        """Returns the length of the path.

        Returns:
            int: The length of the path.
        """
        return len(self._path)

    def __str__(self) -> str:
        """Returns a string representation of the path.

        Returns:
            str: A string representation of the path.
        """

        return "-".join(list(map(str, self._path)))

    def __getitem__(self, idx: Union[int, slice]) -> int:
        """Returns the item at the given index or a slice of items from the path.

        Args:
            idx: An integer index or a slice object.

        Returns:
            int: The item at the given index, or a list of items from the path if a slice is provided.

        Raises:
            IndexError: If the index is out of range.
            TypeError: If the index is not an integer or a slice object.
        """

        if isinstance(idx, int):
            if 0 <= idx < len(self):
                return self._path[idx]
            exception_handler(
                IndexError, "Index out of range", os.EX_DATAERR, self._debug
            )
        elif isinstance(idx, slice):
            return self._path[idx.start : idx.stop : idx.step]
        exception_handler(
            TypeError,
            f"Index must be {int.__name__} or {slice.__name__}, not {type(idx).__name__}",
            os.EX_DATAERR,
            self._debug,
        )

    @property
    def start(self) -> int:
        """Returns the start index of the path.

        Returns:
            int: The start index of the path.
        """
        return self._start

    @property
    def stop(self) -> int:
        """Returns the stop index of the path.

        Returns:
            int: The stop index of the path.
        """
        return self._stop

    @property
    def path(self) -> List[int]:
        """Returns the list of nodes representing the path.

        Returns:
            List[int]: The list of nodes in the path.
        """
        return self._path

    @property
    def sequence(self) -> str:
        """Returns the sequence associated with the path.

        Returns:
            str: The sequence associated with the path.
        """
        return self._sequence


class SVMLogo:
    def __init__(
        self,
        kmers: List[Kmer],
        motifsize: int,
        pruning_threshold: float,
        alphabet: int,
        debug: bool,
    ) -> None:
        self._kmers = kmers
        self._debug = debug
        self._alphabet = alphabet
        self._size = len(self._kmers)
        self._motifsize = motifsize
        self._pruning_threshold = int(len(kmers) * pruning_threshold)
        # construct SVM-logo via greedy procedure
        self._construct_alignment_greedy()

    def _add_vertex_path(
        self, root: int, leaf: int, vids: List[int], sequence: str
    ) -> None:
        p = Path(root, leaf, vids, sequence, self._debug)
        if not hasattr(self, "_paths"):  # initialize _paths
            self._paths = [p]
        else:
            self._paths.append(p)

    def _initialize_weight_matrix(self, pivot: str) -> None:
        if len(pivot) != self._motifsize:
            exception_handler(
                ValueError,
                "Mismatching logo and k-mer length",
                os.EX_DATAERR,
                self._debug,
            )
        size = (
            (self._motifsize + (max(OFFSET) * 2)) * len(ALPHABET[self._alphabet])
        ) + 1
        self._weights = np.zeros((size, size))

    def _update_weights(self, path: List[int]) -> None:
        for i, vid in enumerate(path[:-1]):
            self._weights[vid, path[i + 1]] += 1

    def _initialize_graph_logo(self, pivot: str) -> None:
        self._logo = Graph().as_directed()
        offset = max(OFFSET)
        size = self._motifsize + (offset * 2)
        alphabet_size = len(ALPHABET[self._alphabet])
        for vid in range(size * alphabet_size):
            self._add_vertex(vid)
        for layer, i, j in itertools.product(
            range(size - 1), range(alphabet_size), range(alphabet_size)
        ):
            self._add_edge(
                i + (alphabet_size * layer), j + (alphabet_size * (layer + 1))
            )
        # add stop node and connect it to last logo layer
        self._stop_vid = size * alphabet_size
        self._add_vertex(self._stop_vid)
        # connect last layer and each right offset layer to stop node
        for layer, i in itertools.product(
            range(size - offset - 1, size), range(alphabet_size)
        ):
            self._add_edge((alphabet_size * (layer)) + i, self._stop_vid)
        self._logo.vs["label"] = ALPHABET[self._alphabet] * size + ["*"]
        self._start_vid = alphabet_size * offset
        # initialize paths
        alphamap = ALPHABETMAP[self._alphabet]
        pivot_nodes = [
            self._start_vid + alphamap[nt] + (i * alphabet_size)
            for i, nt in enumerate(pivot)
        ] + [self._stop_vid]
        self._add_vertex_path(pivot_nodes[0], pivot_nodes[-1], pivot_nodes, pivot)
        # initialize weights matrix
        self._initialize_weight_matrix(pivot)
        self._update_weights(pivot_nodes)

    def _add_vertex(self, vid: int) -> None:
        if vid in self._logo.vs.indices:
            exception_handler(
                ValueError, "Forbidden vertex addition", os.EX_DATAERR, self._debug
            )
        self._logo.add_vertex(vid)

    def _add_edge(self, parent: int, child: int) -> None:
        if parent not in self._logo.vs.indices or child not in self._logo.vs.indices:
            exception_handler(
                ValueError, "Forbidden edge addition", os.EX_DATAERR, self._debug
            )
        self._logo.add_edge(parent, child)

    def _match_kmer(self, kmer: Kmer) -> Tuple[str, Path]:
        if not hasattr(self, "_paths"):
            exception_handler(
                ValueError, "Logo sequences undefined", os.EX_DATAERR, self._debug
            )
        # align kmer to kmers in the logo
        matches, seqmatch, pathmatch = -1, None, None
        for p in self._paths:
            if len(p) - 1 != len(kmer):  # skip shorter or longer sequences
                continue
            m, s = self._align_kmer(p.sequence, kmer.kmer)
            if m > matches:
                matches, seqmatch, pathmatch = m, s, p
        assert matches >= 0 and seqmatch is not None and pathmatch is not None
        return seqmatch, pathmatch, matches

    def _align_kmer(self, seq1: str, seq2: str) -> Tuple[int, str]:
        matches_counter_right = [
            self._count_matches(seq1, seq2, start) for start in OFFSET
        ]
        matches_counter_left = [
            self._count_matches(seq2, seq1, start) for start in OFFSET
        ]
        # recover best right and left alignments
        right_align = max(enumerate(matches_counter_right), key=lambda x: sum(x[1]))
        left_align = max(enumerate(matches_counter_left), key=lambda x: sum(x[1]))
        # keep best alignment
        match_nr_right = sum(right_align[1])
        match_nr_left = sum(left_align[1])
        if match_nr_right >= match_nr_left:  # best alignment with offset on the right
            gaps = "".join(["-" for _ in range(right_align[0])])
            seqmatch = "".join(
                [
                    nt if right_align[1][i] else "*"
                    for i, nt in enumerate(seq1[right_align[0] :])
                ]
            )
            return match_nr_right, f"{seqmatch}{gaps}"
        gaps = "".join(
            ["-" for _ in range(left_align[0])]
        )  # best alignment with offset on left
        seqmatch = "".join(
            [
                nt if left_align[1][i] else "*"
                for i, nt in enumerate(seq2[left_align[0] :])
            ]
        )
        return match_nr_left, f"{gaps}{seqmatch}"

    def _count_matches(self, query: str, reference: str, start: int) -> int:
        return [nt == reference[i] for i, nt in enumerate(query[start:])]

    def _extend_weights(self, size: int) -> None:
        if size < 1:
            exception_handler(
                ValueError, "Invalid weight matrix update", os.EX_DATAERR, self._debug
            )
        m, n = self._weights.shape
        self._weights = np.pad(self._weights, ((0, size), (0, size)))
        m2, n2 = self._weights.shape
        assert m + size == m2 and n + size == n2

    def _parents(self, vid: int) -> List[int]:
        parents = [e.tuple[0] for e in self._logo.es if e.tuple[1] == vid]
        if not parents:
            exception_handler(
                ValueError, f"Vertex {vid} is not available", os.EX_DATAERR, self._debug
            )
        return parents

    def _update_labels(self, bvertices: int, bstart_vid: int, kmer: str) -> None:
        labels = self.labels
        for i in range(bvertices):
            labels[self._largest_vid - bvertices + 1 + i] = kmer[bstart_vid + i]
        self._logo.vs["label"] = labels

    def _extend_bubble(
        self, seqmatch: str, start: int, size: int, path: List[int]
    ) -> Tuple[int, int, List[int], int]:
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

    def _close_bubble(
        self, pos: int, size: int, path_current: List[int], path: Path
    ) -> None:
        if pos == size:  # link the bubble to stop node
            self._add_edge(self._largest_vid, self._stop_vid)
            path_current[pos] = self._stop_vid  # add stop vertex to current path
        else:  # link bubble to existing vertex
            self._add_edge(self._largest_vid, path[pos])
            path_current[pos] = path[pos]
        return path_current

    def _modify_logo(
        self, sequence: str, seqmatch: str, path: Path, offset: int, direction: str
    ) -> None:
        alphabet_size = len(ALPHABET[self._alphabet])
        alphamap = ALPHABETMAP[self._alphabet]
        start = self._start_vid
        if direction in [LEFT, RIGHT]:
            start = (
                self._start_vid
                if direction == LEFT
                else self._start_vid + (alphabet_size * offset)
            )
            sequence = sequence[offset:] if direction == LEFT else sequence[:-offset]
        path_current = [
            start + alphamap[nt] + (i * alphabet_size) for i, nt in enumerate(sequence)
        ] + [self._stop_vid]
        path_current = path_current if direction == CENTRAL else path.path
        self._update_weights(path_current)
        if direction == CENTRAL and "*" in seqmatch:  # add the path to paths list
            self._add_vertex_path(
                path_current[0], path_current[-1], path_current, sequence
            )

    def _insert_kmer(self, kmer: Kmer, seqmatch: str, path: Path) -> None:
        offset = seqmatch.count("-")
        direction = CENTRAL
        if offset > 0:
            direction = LEFT if seqmatch.startswith("-") else RIGHT
        self._modify_logo(kmer.kmer, seqmatch, path, offset, direction)

    def _prune_logo(self) -> None:
        # remove edges not visited by any k-mer
        edges = self.edges  # original edges
        incoming_edges = {vid: [] for vid in self.vertices}
        for e in edges:
            incoming_edges[e[1]].append(e)
        for e in edges:
            if self._weights[e[0], e[1]] < self._pruning_threshold:
                self._logo.delete_edges(e)
        # remove disconnected vertices from logo and anchor potential orphans
        connected_vids = {e[i] for e in self.edges for i in [0, 1]}
        alphabet_size = len(ALPHABET[self._alphabet])
        orphans = connected_vids.difference({e[1] for e in self.edges})
        orphans = {
            vid for vid in orphans if vid - self._start_vid >= alphabet_size
        }  # exclude start layer vertices
        for orphan in orphans:
            orphan_weights = [self._weights[e[0], e[1]] for e in incoming_edges[orphan]]
            # anchor orphan node with best incoming edge
            parent, child = incoming_edges[orphan][
                orphan_weights.index(max(orphan_weights))
            ]
            self._add_edge(parent, child)
            connected_vids.add(parent)
        self._logo.delete_vertices(list(set(self.vertices).difference(connected_vids)))

    def _construct_alignment_greedy(self) -> None:
        pivot = self._kmers[0].kmer  # recover pivot sequence
        pivot_weight = self._kmers[0].score
        # initialize graph logo
        self._initialize_graph_logo(pivot)
        # align kmers to current logo
        for kmer in tqdm(self._kmers[1:]):
            seqmatch, path, matches = self._match_kmer(kmer)
            if matches < self._motifsize:
                kmer_rc = Kmer(
                    reverse_complement(kmer.kmer, self._alphabet), kmer.score
                )
                seqmatch_rc, path_rc, matches_rc = self._match_kmer(kmer_rc)
                if matches < matches_rc:
                    seqmatch, path, kmer, matches = (
                        seqmatch_rc,
                        path_rc,
                        kmer_rc,
                        matches_rc,
                    )
            if kmer.score > 0 or matches > self._motifsize / 2:
                self._insert_kmer(kmer, seqmatch, path)
        self._prune_logo()  # prune logo
        # for p in self._paths:
        #     print(p)
        # print(self.vertices)

    def _write_dot(self, logofile: str) -> str:
        palette = PALETTE[self._alphabet]  # current logo color palette
        with open(logofile, mode="w") as outfile:
            self._logo_file_content(outfile, palette)
        assert os.stat(logofile).st_size > 0
        return logofile

    def _logo_file_content(self, outfile: TextIOWrapper, palette: Dict[str, str]):
        # write file header comment
        outfile.write(
            f"/* SVM-Logo v{__version__} -- logo {outfile} created on {datetime.datetime.now()} */"
        )
        if not self._logo.is_dag():
            exception_handler(ValueError, "Unsolvable logo", os.EX_DATAERR, self._debug)
        # ---> start data writing <---
        outfile.write('digraph {\n\trankdir="LR"\n')  # horizontal ranking orientation
        for i, vid in enumerate(self.vertices):
            label = self.labels[i]
            color = palette[label]  # color logo letters
            # TODO: set fontsize according to RE
            # fontsize = 20
            vertex_attributes = (
                f"\t{vid} [\n\t\tname={vid}\n\t\tname={vid}\n"
                f'\t\tlabel="{label}"\n\t\tstyle=filled\n'
                f'\t\tcolor="white"\n\t\tfontcolor="{color}"\n'
                f'\t\tfontname="Arial"\n\t\tfontsize=20\n\t];\n'  # set font style
            )  # close letter field
            outfile.write(vertex_attributes)
        for e in self.edges:
            start, stop = e
            # TODO: set edge weight according to RE
            # weight = 5
            outfile.write(f"\t{start} -> {stop} [penwidth=1];\n")  # close edge field
        outfile.write("}\n")  # close logo file

    def display(self, outfile: str):
        logofile = self._write_dot(outfile)

    @property
    def vertices(self) -> List[int]:
        return self._logo.vs.indices

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return [e.tuple for e in self._logo.es]

    @property
    def edge_map(
        self,
    ) -> Dict[int, Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
        edge_map = {vid: ([], []) for vid in self.vertices}
        for e in self.edges:
            edge_map[e[0]][1].append(e)  # outcoming edges
            edge_map[e[1]][0].append(e)  # incoming edges
        return edge_map

    @property
    def labels(self) -> List[str]:
        return self._logo.vs["label"]
