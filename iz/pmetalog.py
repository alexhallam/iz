import numpy as np
from .support import newtons_method_metalog

def pmetalog(m, q, term=3):
    """Generate probabilities with user specified quantiles from a fitted metalog object.
        Generated using user specified number of terms.
        Quantiles are generated using a Newton's Method approximation.

    Args:
        m (:obj:`metalog`): A fitted metalog object.

        q (:obj:`list` | `numpy.ndarray`): Quantiles to return probabilities values for.

        term (:obj:`int`, optional): Number of metalog terms to use when generating probabilities.
          - strictly >= 2
          - must be in range [m.term_lower_bound, m.term_limit]
          - Default: 3

    Returns:
        (:obj:`list`): len(q) list of probabilities from fitted metalog.

    """
    valid_terms = m.output_dict['model_data']['coeffs'].numpy().flatten()
    term = len(valid_terms)

    if (type(q) != list) and (type(q) != np.ndarray):
        raise TypeError("Error: input q must be a list or numpy array")
    if not isinstance(q, (int, float, complex)) and not all(
        isinstance(x, (int, float, complex)) for x in q
    ):
        raise TypeError("Error: all elements in q must be numeric")
    if (
        #(term in valid_terms) != True
        type(term) != int
        or (term < 2)
        #or ((term % 1) != 0)
    ):
        raise TypeError(
            "Error: term must be a single positive numeric interger contained in the metalog object. Available "
            "terms are: " + " ".join(map(str, valid_terms))
        )

    qs = list(map(lambda qi: newtons_method_metalog(q=qi, m=m, term=term), q))
    return qs
