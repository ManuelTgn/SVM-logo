"""
"""

DNA = 0
RNA = 1
AMINOACIDS = 2

OFFSET = [-3, -2, -1, 0, 1, 2, 3]

ALPHABET = {
    DNA: ["A", "C", "G", "T", "N"],
    RNA: ["A", "C", "G", "U", "N"],
}

RC = {
    DNA: {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"},
    RNA: {"A": "U", "C": "G", "G": "C", "U": "A", "N": "N"}
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


