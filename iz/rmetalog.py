import numpy as np
import pandas as pd

def rmetalog(m, n=1, generator="rand"):
    """Take n random draws from fitted metalog m using specified number of terms.
        Uses specified random seed.

    Args:
        m (:obj:`metalog`): A fitted metalog object.

        n (:obj:`int`, optional): Number of random draws to take from fitted metalog.
          - strictly >= 1
          - Default: 1

        term (:obj:`int`, optional): Number of metalog terms to use when making random draws.
          - strictly >= 2
          - must be in range [m.term_lower_bound, m.term_limit]
          - Default: 2

        generator (:obj:`str`, optional): String that is used to specify the random number generator.
          - must be in set ('rand','hdr')
            * 'rand' uses `np.random.rand`, results are random each time
            * 'hdr' uses Hubbard Decision Research (HDR) random number generator, results are repeatable
          - Default: 'rand'

    Returns:
        (:obj:`numpy.ndarray`): n length numpy array of random draws from fitted metalog.

    """
    m = m.output_dict
    valid_terms = m['model_data']['coeffs'].numpy().flatten()
    print(valid_terms)
    #valid_terms = np.asarray(m["params"]["terms"])
    valid_terms_printout = " ".join(str(t) for t in valid_terms)
    term = len(valid_terms)
    if (type(n) != int) or (n < 1) or ((n % 1) != 0):
        raise TypeError("Error: n must be a positive numeric interger")
    if (
        (type(term) != int)
        or (term < 2)
        or ((term % 1) != 0)
        #or not (term in valid_terms)
    ):
        raise TypeError(
            "Error: term must be a single positive numeric interger contained "
            "in the metalog object. Available terms are: " + valid_terms_printout
        )

    if generator == "hdr":
        x_arr = np.arange(1, n + 1)
        v_index = np.random.randint(80000)

        def hdrgen(pm_index):
            return (
                np.mod(
                    (
                        (
                            np.mod(
                                (v_index + 1000000)
                                ^ 2 + (v_index + 1000000) * (pm_index + 10000000),
                                99999989,
                            )
                        )
                        + 1000007
                    )
                    * (
                        (
                            np.mod(
                                (pm_index + 10000000)
                                ^ 2
                                + (pm_index + 10000000)
                                * (
                                    np.mod(
                                        (v_index + 1000000)
                                        ^ 2
                                        + (v_index + 1000000) * (pm_index + 10000000),
                                        99999989,
                                    )
                                ),
                                99999989,
                            )
                        )
                        + 1000013
                    ),
                    2147483647,
                )
                + 0.5
            ) / 2147483647

        vhdrgen = np.vectorize(hdrgen)
        x = vhdrgen(x_arr)

    else:
        x = np.random.rand(n)

    Y = pd.DataFrame(np.array([np.repeat(1, n)]).T, columns=["y1"])

    # Construct initial Y Matrix values
    Y["y2"] = np.log(x / (1 - x))
    if term > 2:
        Y["y3"] = (x - 0.5) * Y["y2"]
    if term > 3:
        Y["y4"] = x - 0.5

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
    #a = m["A"][amat].iloc[0:(term)].to_frame()
    #s = np.dot(Y, a)
    s = np.dot(Y, valid_terms)

    if m["params"]["boundedness"] == "sl":
        s = m["params"]["bounds"][0] + np.exp(s)

    if m["params"]["boundedness"] == "su":
        s = m["params"]["bounds"][1] - np.exp(-(s))

    if m["params"]["boundedness"] == "b":
        s = (m["params"]["bounds"][0] + (m["params"]["bounds"][1]) * np.exp(s)) / (
            1 + np.exp(s)
        )

    return s
