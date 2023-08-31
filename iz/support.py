import pandas as pd
import numpy as np
import torch

def pdfMetalog(a, y, t, bounds=[], boundedness="u"):
    """Estimates the metalog pdf given the a coefficients and percentiles found using the specified metalog.fit_method attribute.
    Called during metalog.__init__ method call if `fit_method`='MLE'.
    Called during pdf_quantile_builder method call.

    Args:
      a (:obj: `numpy.ndarray` of type float): Array of a coefficients found by fitting metalog distribution using the `fit_method` parameter.

      y (:obj: `numpy.ndarray` of type float): Array of bin widths specified for `a` parameter

      t (:obj: `int`): The upper limit of the range of metalog terms to use to fit the data.
        - metalog.term_limit attribute
        - in range [3,30]

      bounds (:obj: `list`, optional): Upper and lower limits to filter the data with before calculating metalog quantiles/pdfs.
          - should be set in conjunction with the `boundedness` parameter
          - Default: [0,1]

      boundedness (:obj: `str`, optional): String that is used to specify the type of metalog to fit.
          - must be in set ('u','sl','su','b')
          - Default: 'u'
              * Fits an unbounded metalog
              * If `boundedness` parameter != 'u' we must calculate the metalog quantiles using an unbounded metalog, via the `quantileMetalog` method.
          - 'sl' fits a strictly lower bounded metalog
              * len(bounds) must == 1
          - 'su' fits a strictly upper bounded metalog
              * len(bounds) must == 1
          - 'b' fits a upper/lower bounded metalog
              * len(bounds) must == 2
              * bounds[1] must be > bounds[0]

    Returns:
      x: (:obj: `numpy.ndarray` of type float): Array of metalog pdf values.

    """
    if y <= 0:
        y = 0.00001

    if y >= 1:
        y = 0.99999

    d = y * (1 - y)
    f = y - 0.5
    l = np.log(y / (1 - y))

    # Initiate pdf

    # For the first three terms
    x = a[1] / d
    if len(a) > 2 and a[2] != 0:
        x = x + a[2] * ((f / d) + l)

    # For the fourth term
    if t > 3:
        x = x + a[3]

    # Initalize some counting variables
    e = 1
    o = 1

    # For all other terms greater than 4
    if t > 4:
        for i in range(5, t + 1):
            if (i % 2) != 0:
                # iff odd
                x = x + ((o + 1) * a[i - 1] * f ** o)
                o = o + 1

            if (i % 2) == 0:
                # iff even
                x = x + a[i - 1] * (((f ** (e + 1)) / d) + (e + 1) * (f ** e) * l)
                e = e + 1

    # Some change of variables here for boundedness
    x = x ** (-1)

    if boundedness != "u":
        M = quantileMetalog(a, y, t, bounds=bounds, boundedness="u")

    if boundedness == "sl":
        x = x * np.exp(-M)

    if boundedness == "su":
        x = x * np.exp(M)

    if boundedness == "b":
        x = (x * (1 + np.exp(M)) ** 2) / ((bounds[1] - bounds[0]) * np.exp(M))

    if x <= 0:
        x = 0.00001
    # print(str(x) + " zoop")

    return x


def quantileMetalog(a, y, t, bounds=[], boundedness="u"):
    """Estimates the metalog quantiles given the a coefficients and percentiles found using the specified metalog.fit_method attribute.
    Called during metalog.__init__ method call if `fit_method`='MLE'.
    Called during pdf_quantile_builder method call.

    Args:
      a (:obj: `numpy.ndarray` of type float): Array of a coefficients found by fitting metalog distribution using the `fit_method` parameter.

      y (:obj: `numpy.ndarray` of type float): Array of bin widths specified for `a` parameter

      t (:obj: `int`): The upper limit of the range of metalog terms to use to fit the data.
        - metalog.term_limit attribute
        - in range [3,30]

      bounds (:obj: `list`, optional): Upper and lower limits to filter the data with before calculating metalog quantiles/pdfs.
          - should be set in conjunction with the `boundedness` parameter
          - Default: [0,1]

      boundedness (:obj: `str`, optional): String that is used to specify the type of metalog to fit.
          - must be in set ('u','sl','su','b')
          - Default: 'u'
              * Fits an unbounded metalog
          - 'sl' fits a strictly lower bounded metalog
              * len(bounds) must == 1
          - 'su' fits a strictly upper bounded metalog
              * len(bounds) must == 1
          - 'b' fits a upper/lower bounded metalog
              * len(bounds) must == 2
              * bounds[1] must be > bounds[0]

    Returns:
      x: (:obj: `numpy.ndarray` of type float): Array of metalog quantile values.

    """
    if y <= 0:
        y = 0.00001

    if y >= 1:
        y = 0.99999
    # Some values for calculation
    f = y - 0.5
    l = np.log(y / (1 - y))

    # For the first three terms
    x = a[0] + a[1] * l
    if t > 2:
        x = x + a[2] * f * l

    # For the fourth term
    if t > 3:
        x = x + a[3] * f

    # Some tracking variables
    o = 2
    e = 2

    # For all other terms greater than 4
    if t > 4:
        for i in range(5, t + 1):
            if (i % 2) == 0:
                x = x + a[i - 1] * f ** e * l
                e = e + 1
            if (i % 2) != 0:
                x = x + a[i - 1] * f ** o
                o = o + 1

    if boundedness == "sl":
        x = bounds[0] + np.exp(x)

    if boundedness == "su":
        x = bounds[1] - np.exp(-x)

    if boundedness == "b":
        x = (bounds[0] + bounds[1] * np.exp(x)) / (1 + np.exp(x))

    return x


def diffMatMetalog(term_limit, step_len):
    """TODO: write docstring"""
    y = np.arange(step_len, 1, step_len)
    Diff = np.array([])

    for i in range(0, (len(y))):
        d = y[i] * (1 - y[i])
        f = y[i] - 0.5
        l = np.log(y[i] / (1 - y[i]))

        # Initiate pdf
        diffVector = 0

        # For the first three terms
        x = 1 / d
        diffVector = [diffVector, x]

        if term_limit > 2:
            diffVector.append((f / d) + l)

        # For the fourth term
        if term_limit > 3:
            diffVector.append(1)

        # Initalize some counting variables
        e = 1
        o = 1

        # For all other terms greater than 4
        if term_limit > 4:
            for i in range(5, (term_limit + 1)):
                if (i % 2) != 0:
                    # iff odd
                    diffVector.append((o + 1) * f ** o)
                    o = o + 1

                if (i % 2) == 0:
                    # iff even
                    diffVector.append(((f ** (e + 1)) / d) + (e + 1) * (f ** e) * l)
                    e = e + 1
        if np.size(Diff) == 0:
            Diff = diffVector
        else:
            Diff = np.vstack((Diff, diffVector))

    Diff_neg = -1 * (Diff)
    new_Diff = np.hstack((Diff[:, [0]], Diff_neg[:, [0]]))

    for c in range(1, (len(Diff[1, :]))):
        new_Diff = np.hstack((new_Diff, Diff[:, [c]]))
        new_Diff = np.hstack((new_Diff, Diff_neg[:, [c]]))

    new_Diff = pd.DataFrame(data=new_Diff)

    return new_Diff


def newtons_method_metalog(m, q, term, bounds=None, boundedness=None):
    """TODO: write docstring"""
    # a simple newtons method application
    m = m.output_dict
    if bounds == None:
        bounds = [x if (type(x) != torch.Tensor) else x.numpy().astype(int) for x in m["params"]["bounds"]]
        #bounds = m["params"]["bounds"]
    if boundedness == None:
        boundedness = m["params"]["boundedness"]

    # if m is metalog
    try:
        m = m.output_dict
        #avec = "a" + str(term)
        a = m['model_data']['coeffs'].numpy().flatten()
        #a = m["A"][avec]
    except:
        a = m

    # TODO there should be setters for at least some of these hyperparameters
    alpha_step = 0.5
    err = 1e-10
    temp_err = 0.1
    y_now = 0.5

    i = 1
    a = m['model_data']['coeffs'].numpy().flatten()
        #a = m["A"][avec]
    while temp_err > err:
        frist_function = quantileMetalog(a, y_now, term, bounds, boundedness) - q
        derv_function = pdfMetalog(a, y_now, term, bounds, boundedness)
        y_next = y_now - alpha_step * (frist_function * derv_function)
        temp_err = abs((y_next - y_now))

        if y_next > 1:
            y_next = 0.99999

        if y_next < 0:
            y_next = 0.000001

        y_now = y_next
        i = i + 1

        if i > 10000:
            raise StopIteration(
                "Approximation taking too long, quantile value: "
                + str(q)
                + " is to far from distribution median. Try plot() to see distribution."
            )

    return y_now


def pdfMetalog_density(m, t, y):
    m = m.output_dict
    #avec = "a" + str(t)
    #a = m["A"][avec]
    a = m['model_data']['coeffs'].numpy().flatten()
    bounds = m["params"]["bounds"]
    boundedness = m["params"]["boundedness"]

    d = y * (1 - y)
    f = y - 0.5
    l = np.log(y / (1 - y))

    # Initiate pdf

    # For the first three terms
    x = a[1] / d
    if a[2] != 0:
        x = x + a[2] * ((f / d) + l)

    # For the fourth term
    if t > 3:
        x = x + a[3]

    # Initalize some counting variables
    e = 1
    o = 1

    # For all other terms greater than 4
    if t > 4:
        for i in range(5, t + 1):
            if (i % 2) != 0:
                # iff odd
                x = x + ((o + 1) * a[i - 1] * f ** o)
                o = o + 1

            if (i % 2) == 0:
                # iff even
                x = x + a[i - 1] * (((f ** (e + 1)) / d) + (e + 1) * (f ** e) * l)
                e = e + 1

    # Some change of variables here for boundedness

    x = x ** (-1)

    if boundedness != "u":
        M = quantileMetalog(a, y, t, bounds=bounds, boundedness="u")

    if boundedness == "sl":
        x = x * np.exp(-M)

    if boundedness == "su":
        x = x * np.exp(M)

    if boundedness == "b":
        x = (x * (1 + np.exp(M)) ** 2) / ((bounds[1] - bounds[0]) * np.exp(M))

    return x
