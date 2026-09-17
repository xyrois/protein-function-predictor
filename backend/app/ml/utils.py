from collections import Counter

# Approximate molecular weights of amino acids in Daltons
AA_WEIGHTS = {
    'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
    'E': 147.1, 'Q': 146.2, 'G': 75.1,  'H': 155.2, 'I': 131.2,
    'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
    'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
}

def calculate_sequence_stats(sequence: str) -> dict:
    """Calculate sequence length, molecular weight, and charge profile."""
    seq_len = len(sequence)
    # Deduct water (18.015 Da) for each peptide bond
    mw_da = sum(AA_WEIGHTS.get(aa, 110.0) for aa in sequence) - (seq_len - 1) * 18.015
    mw_kda = round(mw_da / 1000.0, 2)

    counts = Counter(sequence)
    pos_charge = counts.get('K', 0) + counts.get('R', 0) + counts.get('H', 0)
    neg_charge = counts.get('D', 0) + counts.get('E', 0)

    if pos_charge > neg_charge:
        pi_est = "Basic (pI > 7.0)"
    elif neg_charge > pos_charge:
        pi_est = "Acidic (pI < 7.0)"
    else:
        pi_est = "Neutral (pI ~ 7.0)"

    return {
        "length": seq_len,
        "molecular_weight_approx_kda": max(mw_kda, 0.1),
        "isoelectric_point_estimate": pi_est
    }