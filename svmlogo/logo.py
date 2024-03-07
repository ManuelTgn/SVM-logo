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
    DNA,
    RNA,
    reverse_complement,
    transform,
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

    # vertex functions
    def _add_vertex(self, vid: int) -> None:
        """
        Adds a vertex to the logo graph.

        Args:
            vid: An integer representing the vertex ID to be added.

        Returns:
            None
        """

        if vid in self._logo.vs.indices:
            exception_handler(
                ValueError, "Forbidden vertex addition", os.EX_DATAERR, self._debug
            )
        self._logo.add_vertex(vid)

    def _add_vertices(self, num_vertices: int) -> None:
        """
        Adds vertices to the SVMLogo object.

        Args:
            num_vertices: The number of vertices to add.

        Returns:
            None
        """

        for vid in range(num_vertices):
            self._add_vertex(vid)

    # edge functions
    def _add_edge(self, parent: int, child: int) -> None:
        """
        Adds an edge between two vertices in the logo graph.

        Args:
            parent: An integer representing the parent vertex ID.
            child: An integer representing the child vertex ID.

        Returns:
            None
        """

        if parent not in self._logo.vs.indices or child not in self._logo.vs.indices:
            exception_handler(
                ValueError, "Forbidden edge addition", os.EX_DATAERR, self._debug
            )
        self._logo.add_edge(parent, child)

    def _insert_edges_layers(self, size: int, alphabet_size: int) -> None:
        """
        Inserts edges between layers in the SVMLogo object.

        Args:
            size: The size of the logo.
            alphabet_size: The size of the alphabet.

        Returns:
            None
        """

        for layer in range(size - 1):
            for i, j in itertools.product(range(alphabet_size), repeat=2):
                self._add_edge(
                    i + (alphabet_size * layer), j + (alphabet_size * (layer + 1))
                )

    def _connect_last_layer_to_stop(
        self, size: int, offset: int, alphabet_size: int
    ) -> None:
        """
        Connects the last layer of vertices in the SVMLogo object to the stop node.

        Args:
            size: The size of the logo.
            offset: The offset value.
            alphabet_size: The size of the alphabet.

        Returns:
            None
        """

        for layer, i in itertools.product(
            range(size - offset - 1, size), range(alphabet_size)
        ):
            self._add_edge((alphabet_size * layer) + i, self._stop_vid)

    # path functions
    def _add_vertex_path(
        self, root: int, leaf: int, vids: List[int], sequence: str
    ) -> None:
        """
        Adds a path to the list of paths in the SVMLogo object.

        Args:
            root: The start index of the path.
            leaf: The stop index of the path.
            vids: A list of integers representing the nodes in the path.
            sequence: A string representing the sequence associated with the path.

        Returns:
            None
        """

        p = Path(root, leaf, vids, sequence, self._debug)
        if not hasattr(self, "_paths"):  # initialize _paths
            self._paths = [p]
        else:
            self._paths.append(p)

    # weight matrix functions
    def _initialize_weight_matrix(self, pivot: str) -> None:
        """
        Initializes the weight matrix for the SVMLogo object.

        Args:
            pivot: The pivot string used to determine the size of the weight matrix.

        Returns:
            None
        """

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
        """
        Updates the weight matrix based on the given path.

        Args:
            path: A list of integers representing the nodes in the path.

        Returns:
            None
        """

        for i, vid in enumerate(path[:-1]):
            self._weights[vid, path[i + 1]] += 1

    # score array functions
    def _initialize_score_array(self) -> None:
        """
        Initializes the score array for the SVMLogo object.

        Returns:
            None
        """

        size = (
            (self._motifsize + (max(OFFSET) * 2)) * len(ALPHABET[self._alphabet])
        ) + 1
        self._scores = np.array([0.0] * size)

    def _initialize_weight_score(
        self, pivot: str, pivot_path: List[int], score: float
    ) -> None:
        """
        Initializes the weight matrix and score array for the SVMLogo object based
        on the given pivot, pivot path, and score.

        Args:
            pivot: A string representing the pivot.
            pivot_path: A list of integers representing the pivot path.
            score: A float representing the score.

        Returns:
            None
        """

        self._initialize_weight_matrix(pivot)  # initialize weight matrix
        self._update_weights(pivot_path)
        self._initialize_score_array()  # initialize score array
        self._update_scores(pivot_path, score)

    def _update_scores(self, path: List[int], score: float) -> None:
        """
        Updates the score array based on the given path and score.

        Args:
            path: A list of integers representing the nodes in the path.
            score: The score value to update the array with.

        Returns:
            None
        """

        for vid in path[:-1]:
            if score > self._scores[vid]:
                self._scores[vid] = score

    # logo initialiazation functions
    def _get_pivot_nodes(self, pivot: str, alphabet_size: int) -> List[int]:
        """
        Returns pivot path nodes for the SVMLogo object.

        Args:
            pivot: A string representing the pivot.
            alphabet_size: The size of the alphabet.

        Returns:
            List[int]: A list of integers representing the pivot nodes.
        """

        alphamap = ALPHABETMAP[self._alphabet]
        return [
            self._start_vid + alphamap[c] + (i * alphabet_size)
            for i, c in enumerate(pivot)
        ] + [self._stop_vid]

    def _initialize_graph_logo(self, pivot: str, pivot_score: float) -> None:
        """
        Initializes the logo graph for the SVMLogo object based on the given pivot
        and pivot score.

        Args:
            pivot: A string representing the pivot.
            pivot_score: A float representing the score of the pivot.

        Returns:
            None
        """

        self._logo = Graph().as_directed()  # initialize logo
        offset = max(OFFSET)  # offset size (left and right offsets)
        size = self._motifsize + (offset * 2)  # motif size considering the offsets
        alphabet_size = len(ALPHABET[self._alphabet])  # recover alphabet size
        # add vertices to logo
        num_vertices = size * alphabet_size  # total number of logo vertices
        self._add_vertices(num_vertices)
        # insert edges between layers' vertices
        self._insert_edges_layers(size, alphabet_size)
        # define stop node and connect it to last layers
        self._stop_vid = num_vertices
        self._add_vertex(self._stop_vid)
        self._connect_last_layer_to_stop(size, offset, alphabet_size)
        # set vertex labels and names
        self._logo.vs["label"] = ALPHABET[self._alphabet] * size + ["*"]
        self._logo.vs["name"] = list(range(num_vertices))
        # initialize logo paths
        self._start_vid = alphabet_size * offset
        pivot_nodes = self._get_pivot_nodes(pivot, alphabet_size)
        self._add_vertex_path(pivot_nodes[0], pivot_nodes[-1], pivot_nodes, pivot)
        # initialize weight matrix and score array
        self._initialize_weight_score(pivot, pivot_nodes, pivot_score)

    # k-mers alignment functions
    def _match_kmer(self, kmer: Kmer) -> Tuple[str, Path]:
        """
        Matches a k-mer to the logo sequences and returns the best matching sequence,
        path, and number of matches.

        Args:
            kmer: A Kmer object representing the k-mer to be matched.

        Returns:
            Tuple[str, Path, int]: A tuple containing the best matching sequence, path,
            and number of matches.
        """

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

    def _count_matches(self, query: str, reference: str, start: int) -> int:
        """
        Counts the number of matches between a query sequence and a reference
        sequence starting from a given position.

        Args:
            query: The query sequence.
            reference: The reference sequence.
            start: The starting position for the comparison.

        Returns:
            int: The number of matches between the query and reference sequences.
        """

        return [nt == reference[i] for i, nt in enumerate(query[start:])]

    def _align_kmer(self, seq1: str, seq2: str) -> Tuple[int, str]:
        """
        Aligns two sequences and returns the best alignment with the number of matches.

        Args:
            seq1: The first sequence to align.
            seq2: The second sequence to align.

        Returns:
            Tuple[int, str]: A tuple containing the number of matches and the best alignment.
        """

        matches_counter_right = [
            self._count_matches(seq1, seq2, start) for start in OFFSET
        ]
        matches_counter_left = [
            self._count_matches(seq2, seq1, start) for start in OFFSET
        ]
        # recover best right and left alignments
        right_align = max(matches_counter_right, key=sum)
        left_align = max(matches_counter_left, key=sum)
        # keep best alignment
        match_nr_right = sum(right_align)
        match_nr_left = sum(left_align)
        if match_nr_left < match_nr_right:  # besta alignment with offset on right
            offset = matches_counter_right.index(right_align)
            seqmatch = "".join(
                c if match else "*" for match, c, in zip(right_align, seq1[offset:])
            )
            seqmatch += "-" * offset
            return match_nr_right, seqmatch
        offset = matches_counter_left.index(
            left_align
        )  # best alignment with offset on left
        seqmatch = "".join(
            c if match else "*" for match, c in zip(left_align, seq2[offset:])
        )
        seqmatch = ("-" * offset) + seqmatch
        return match_nr_left, seqmatch

    # logo greedy alignment construction functions
    def _update_logo(
        self,
        sequence: str,
        score: float,
        seqmatch: str,
        path: Path,
        offset: int,
        direction: str,
    ) -> None:
        """
        Modifies the logo based on the given sequence, score, alignment, path, offset,
        and direction.

        Args:
            sequence: A string representing the sequence to modify the logo with.
            score: A float representing the score of the sequence.
            seqmatch: A string representing the aligned sequence.
            path: A Path object representing the path in the logo graph.
            offset: An integer representing the offset value.
            direction: A string representing the direction of the modification.

        Returns:
            None
        """

        alphabet_size = len(ALPHABET[self._alphabet])
        alphamap = ALPHABETMAP[self._alphabet]
        start = (
            self._start_vid + (alphabet_size * offset)
            if direction == RIGHT
            else self._start_vid
        )
        sequence = (
            sequence[offset:]
            if direction == LEFT
            else sequence[:-offset]
            if direction == RIGHT
            else sequence
        )
        path_current = [
            start + alphamap[c] + (i * alphabet_size) for i, c in enumerate(sequence)
        ] + [self._stop_vid]
        # update weight and score matrices according to highest matching path
        self._update_weights(path_current if direction == CENTRAL else path.path)
        self._update_scores(path_current if direction == CENTRAL else path.path, score)
        if direction == CENTRAL and "*" in seqmatch:  # diverging path
            self._add_vertex_path(
                path_current[0], path_current[-1], path_current, sequence
            )

    def _insert_kmer(self, kmer: Kmer, seqmatch: str, path: Path) -> None:
        """
        Inserts a k-mer into the logo graph based on the alignment information.

        Args:
            kmer: A Kmer object representing the k-mer to be inserted.
            seqmatch: A string representing the aligned sequence.
            path: A Path object representing the path in the logo graph.

        Returns:
            None
        """

        offset = seqmatch.count("-")
        direction = CENTRAL
        if offset > 0:
            direction = LEFT if seqmatch.startswith("-") else RIGHT
        self._update_logo(kmer.kmer, kmer.score, seqmatch, path, offset, direction)

    def _prune_logo(self) -> None:
        """
        Prunes the logo by removing edges and vertices that are not visited by any k-mer.

        Returns:
            None
        """

        # remove edges not visited by any k-mer
        incoming_edges = {vid: [] for vid in self.vertices}
        for e in self.edges:
            incoming_edges[e[1]].append(e)
        self._logo.delete_edges(
            {
                e
                for e in self.edges
                if self._weights[e[0], e[1]] < self._pruning_threshold
            }
        )
        # remove disconnected nodes and anchor potential orphans
        connected_nodes = {e[i] for e in self.edges for i in [0, 1]}
        alphabet_size = len(ALPHABET[self._alphabet])  # alphabet size
        orphans = {
            vid
            for vid in connected_nodes
            if vid not in {e[1] for e in self.edges}
            and vid - self._start_vid >= alphabet_size
        }
        for orphan in orphans:  # anchor orphan nodes
            parent, child = max(
                incoming_edges[orphan], key=lambda e: self._weights[e[0], e[1]]
            )
            self._add_edge(parent, child)  # restore edge
            connected_nodes.add(parent)
        # remove edges connected to stop node and the node itself
        stop_edges = {e for e in self.edges if e[1] == self._stop_vid}
        self._logo.delete_edges(stop_edges)
        connected_nodes.remove(self._stop_vid)
        self._logo.delete_vertices(set(self.vertices).difference(connected_nodes))

    def _reverse_logo(self) -> None:
        """
        Compute reverse complement logo by reversing the order of vertices and labels.

        Returns:
            None
        """

        if not hasattr(self, "_logo"):
            exception_handler(
                AttributeError,
                "Cannot compute reverse logo",
                os.EX_DATAERR,
                self._debug,
            )
        size = len(self.vertices)  # logo size
        reversed_vertices = [size - 1 - i for i in range(size)]
        self._logo_rc = Graph(
            edges=[
                (reversed_vertices[e[1]], reversed_vertices[e[0]]) for e in self.edges
            ],
            directed=True,
        )
        self._logo_rc.vs["label"] = [
            reverse_complement(c, self._alphabet) for c in self.labels[::-1]
        ]

    def _construct_alignment_greedy(self) -> None:
        """
        Constructs the alignment using a greedy approach based on the k-mers.

        The alignment is constructed by iteratively aligning each k-mer to the current logo.
        The alignment is based on the number of matches between the k-mer and the logo sequences.
        The alignment is pruned and, if the alphabet is DNA or RNA, reversed.

        Returns:
            None
        """

        pivot = self._kmers[0].kmer  # recover pivot sequence
        pivot_score = self._kmers[0].score
        # initialize graph logo
        self._initialize_graph_logo(pivot, pivot_score)
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
        if self._alphabet in [DNA, RNA]:
            self._reverse_logo()  # compute reverse complement if possible

    def _write_vertex_data(
        self,
        outfile: TextIOWrapper,
        vid: int,
        label: int,
        color: str,
        size: int,
        reverse: bool,
    ) -> None:
        """
        Writes the vertex data to the logo file.

        Args:
            outfile: A TextIOWrapper object representing the output file.
            vid: An integer representing the vertex ID.
            label: An integer representing the vertex label.
            color: A string representing the vertex color.
            size: An integer representing the size of the logo.
            reverse: A boolean indicating whether the logo is reversed.

        Returns:
            None
        """

        idx = (
            self.names[size - 1 - vid] if reverse else self.names[vid]
        )  # recover vertex name
        # compute fontsize according to character weight
        fontsize = transform(
            self._scores[idx], 20, 70, np.max(self._scores), np.min(self._scores)
        )
        # write vertex attributes to logo file
        vertex_attributes = {
            "vid": vid,
            "label": label,
            "fontcolor": color,
            "fontsize": fontsize,
        }
        outfile.write(
            '\t{vid} [\n\t\tname={vid}\n\t\tlabel="{label}"\n\t\tstyle=filled\n'
            '\t\tcolor="white"\n\t\tfontcolor="{fontcolor}"\n'
            '\t\tfontname="Arial"\n\t\tfontsize={fontsize}\n\t];\n'.format(
                **vertex_attributes
            )
        )

    # logo display functions
    def _write_logo_file_content(
        self, outfile: TextIOWrapper, palette: Dict[str, str], reverse: bool
    ) -> None:
        """
        Writes the content of the logo graph to a DOT file.

        Args:
            outfile: A TextIOWrapper object representing the output file.
            palette: A dictionary mapping labels to colors.
            reverse: A boolean indicating whether to write the reverse complement logo.

        Returns:
            None
        """

        if reverse and not hasattr(self, "_logo_rc"):
            exception_handler(
                AttributeError,
                "Reverse complement logo not available",
                os.EX_DATAERR,
                self._debug,
            )
        logo = self._logo_rc if reverse else self._logo
        if not logo.is_dag():
            exception_handler(ValueError, "Unsolvable logo", os.EX_DATAERR, self._debug)
        # write header comments
        outfile.write(
            f"/* SVM-Logo v{__version__} -- logo {outfile} created on {datetime.datetime.now()} */\n"
        )
        # start writing logo content
        outfile.write(
            'digraph {\n\trankdir="LR"\n'
        )  # force horizontal (left-to-right) ranking orientation
        vertices, labels, edges = (
            (self.vertices_rc, self.labels_rc, self.edges_rc)
            if reverse
            else (self.vertices, self.labels, self.edges)
        )
        size = len(vertices)  # logo size
        for i, vid in enumerate(vertices):  # vertices data
            self._write_vertex_data(
                outfile, vid, labels[i], palette[labels[i]], size, reverse
            )
        for parent, child in edges:  # edges data
            outfile.write(f"\t{parent} -> {child} [penwidth=1];\n")
        outfile.write("}\n")  # close file writing

    def _write_dot(self, logofile: str, reverse: bool) -> str:
        """
        Writes the logo graph to a DOT file.

        Args:
            logofile: The path to the DOT file to write.
            reverse: A boolean indicating whether to reverse the logo graph.

        Returns:
            str: The path to the written DOT file.
        """

        palette = PALETTE[self._alphabet]  # current logo color palette
        with open(logofile, mode="w") as outfile:
            self._write_logo_file_content(outfile, palette, reverse)
        assert os.stat(logofile).st_size > 0
        return logofile

    def display(self, outfile: str, reverse: bool) -> None:
        """
        Displays the logo by writing it to a DOT file and plotting it.

        Args:
            outfile: A string representing the path to the output DOT file.
            reverse: A boolean indicating whether to plot the reverse complement logo.

        Returns:
            None
        """

        if reverse:  # plot reverse complement logo
            logofile_rc = self._write_dot(
                f"{os.path.splitext(outfile)[0]}_rc.dot", True
            )
        logofile = self._write_dot(outfile, False)

    @property
    def vertices(self) -> List[int]:
        """
        Returns the list of vertex IDs in the logo graph.

        Returns:
            List[int]: A list of integers representing the vertex IDs in the logo graph.
        """

        return self._logo.vs.indices

    @property
    def vertices_rc(self) -> List[int]:
        """
        Returns the list of vertex IDs in the reverse complement logo graph.

        Returns:
            List[int]: A list of integers representing the vertex IDs in the reverse
            complement logo graph.
        """

        return self._logo_rc.vs.indices

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """
        Returns the list of edges in the logo graph.

        Returns:
            List[Tuple[int, int]]: A list of tuples representing the edges in the
            logo graph.
        """

        return [e.tuple for e in self._logo.es]

    @property
    def edges_rc(self) -> List[Tuple[int, int]]:
        """
        Returns the list of edges in the reverse complement logo graph.

        Returns:
            List[Tuple[int, int]]: A list of tuples representing the edges in the
            reverse complement logo graph.
        """

        return [e.tuple for e in self._logo_rc.es]

    @property
    def labels(self) -> List[str]:
        """
        Returns the list of labels in the logo graph.

        Returns:
            List[str]: A list of strings representing the labels in the logo graph.
        """

        return self._logo.vs["label"]

    @property
    def labels_rc(self) -> List[str]:
        """
        Returns the list of labels in the reverse complement logo graph.

        Returns:
            List[str]: A list of strings representing the labels in the reverse
            complement logo graph.
        """

        return self._logo_rc.vs["label"]

    @property
    def names(self) -> List[int]:
        """
        Returns the list of names in the logo graph.

        Returns:
            List[int]: A list of integers representing the names in the logo graph.
        """

        return self._logo.vs["name"]
