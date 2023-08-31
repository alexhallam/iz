import numpy as np
import polars as pl
import pandas as pd

def qmetalog(m, y):
    """Generate quantiles with a probability from a fitted metalog object.

    Args:
        m (:obj:`metalog`): A fitted metalog object.

        y (:obj:`list` | `numpy.ndarray`): Probabilities to return quantile values for.

        term (:obj:`int`, optional): Number of metalog terms to use when generating quantiles.
          - strictly >= 2
          - must be in range [m.term_lower_bound, m.term_limit]
          - Default: 3

    Returns:
        (:obj:`numpy.ndarray`): len(q) length numpy array of quantiles from fitted metalog.

    """
    m = m.output_dict
    valid_terms = m['model_data']['coeffs'].numpy().flatten()
    term = len(valid_terms)
    valid_terms_printout = " ".join(str(t) for t in valid_terms)

    if type(y) != list:
        raise TypeError("Error: y must be a list of numeric values")
    y = np.asarray(y)
    if (
        (all(isinstance(x, (int, float, complex)) for x in y)) != True
        or (max(y) >= 1)
        or (min(y) <= 0)
    ):
        raise TypeError(
            "Error: y or all elements in y must be positive numeric values between 0 and 1"
        )
    if (
        (type(term) != int)
        or (term < 2)
        or ((term % 1) != 0)
        #or (term in valid_terms) != True
    ):
        raise TypeError(
            "Error: term must be a single positive numeric integer contained "
            "in the metalog object. Available terms are: " + valid_terms_printout
        )

    Y = pd.DataFrame(np.array([np.repeat(1, len(y))]).T, columns=["y1"])

    # Construct the Y Matrix initial values
    Y["y2"] = np.log(y / (1 - y))
    if term > 2:
        Y["y3"] = (y - 0.5) * Y["y2"]
    if term > 3:
        Y["y4"] = y - 0.5

    # Complete the values through the term limit
    if term > 4:
        for i in range(5, (term + 1)):
            y = "".join(["y", str(i)])
            if i % 2 != 0:
                Y[y] = Y["y4"] ** (i // 2)
            if i % 2 == 0:
                z = "".join(["y", str(i - 1)])
                Y[y] = Y["y2"] * Y[z]

    amat = "".join(["a", str(term)])
    # a = m["A"][amat].iloc[0:(term)].to_frame()
    # s = np.dot(Y, a)
    s = np.dot(Y, valid_terms)

    if m["params"]["boundedness"] == "sl":
        s = m["params"]["bounds"][0] + np.exp(s)

    if m["params"]["boundedness"] == "su":
        s = m["params"]["bounds"][1] - np.exp(-(s))

    if m["params"]["boundedness"] == "b":
        s = (m["params"]["bounds"][0] + (m["params"]["bounds"][1]) * np.exp(s)) / (
            1 + np.exp(s)
        )

    s = s.flatten()
    return s
