import numpy as np
from .support import newtons_method_metalog, pdfMetalog_density

def dmetalog(m, q):
    """Generate density values with user specified quantiles from a fitted metalog object.
        Generated using user specified number of terms.
        Quantiles are generated using a Newton's Method approximation.

    Args:
        m (:obj:`metalog`): A fitted metalog object.

        q (:obj:`list` | `numpy.ndarray`): Quantiles to return density values for.

        term (:obj:`int`, optional): Number of metalog terms to use when generating densities.
          - strictly >= 2
          - must be in range [m.term_lower_bound, m.term_limit]
          - Default: 3

    Returns:
        (:obj:`list`): len(q) list of density values from fitted metalog.

    """
    valid_terms = m.output_dict['model_data']['coeffs'].numpy().flatten()
    #valid_terms = np.asarray(m["params"]["terms"])
    term = len(valid_terms)
    #valid_terms = np.asarray(m.output_dict["Validation"]["term"])

    if (type(q) != list) and (type(q) != np.ndarray):
        raise TypeError("Error: input q must be a list or numpy array.")

    if (
        #(term not in valid_terms)
        #or type(term) != int
        type(term) != int
        or (term < 2)
        #or ((term % 1) != 0)
    ):
        raise TypeError(
            "Error: term must be a single positive numeric interger contained in the metalog object. Available "
            "terms are: " + " ".join(map(str, valid_terms))
        )

    qs = list(map(lambda qi: newtons_method_metalog(q=qi, m=m, term=term), q))
    ds = list(map(lambda yi: pdfMetalog_density(y=yi, m=m, t=term), qs))

    return ds