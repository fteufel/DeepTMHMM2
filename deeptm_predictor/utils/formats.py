"""
Formatting utilities for DeepTM output files.
Handles generation of GFF3 format from predicted region lists.
"""

from typing import List, Tuple

def gff3_format(prot_id: str, seq_len: int, named_regions: List[Tuple[str, int, int]]) -> str:
    """
    Formats predicted regions into a GFF3 compliant string.
    
    Parameters
    ----------
    prot_id : str
        The protein identifier.
    seq_len : int
        Total length of the amino acid sequence.
    named_regions : List[Tuple[str, int, int]]
        List of segments with their biological names, start, and end positions.
        
    Returns
    -------
    str
        GFF3 formatted string for the protein segments.
    """
    
    # Calculate number of Transmembrane Regions
    tm_count = sum(1 for name, _, _ in named_regions if name in ['TMhelix', 'Beta sheet'])
    
    output = [
        f"## {prot_id} Length: {seq_len}",
        f"## {prot_id} Number of predicted TMRs: {tm_count}"
    ]
    
    # Generate GFF rows for each segment
    for name, start, end in named_regions:
        # Sequence name, Source, Type, Start, End, Score, Strand, Phase, Attributes
        output.append(f"{prot_id}\tDeepTM\t{name}\t{start}\t{end}\t.\t.\t.\t.")
    
    return "\n".join(output) + "\n//\n"
