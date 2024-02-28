"""
"""

DNA = 0
RNA = 1
AMINOACIDS = 2
CENTRAL = "central"
RIGHT = "right"
LEFT = "left"


OFFSET = [0, 1, 2, 3]

ALPHABET = {
    DNA: ["A", "C", "G", "T", "N"],
    RNA: ["A", "C", "G", "U", "N"],
}

ALPHABETMAP = {
    DNA: {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4},
    RNA: {"A": 0, "C": 1, "G": 2, "U": 3, "N": 4},
}

RC = {
    DNA: {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"},
    RNA: {"A": "U", "C": "G", "G": "C", "U": "A", "N": "N"},
}

PALETTE = {
    DNA: {
        "A": "#109648",
        "C": "#255c99",
        "G": "#f7b32b",
        "T": "#d62839",
        "N": "#a7a5a4",
        "*": "black",
    },
}


def reverse_complement(sequence: str, alphabet: int) -> str:
    """
    Returns the reverse complement of the given sequence.

    Args:
        sequence (str): The input sequence.
        alphabet (int): The alphabet identifier.

    Returns:
        str: The reverse complement of the input sequence.
    """

    return "".join([RC[alphabet][nt] for nt in sequence[::-1]])
