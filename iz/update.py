
def update(m, new_data, penalty=None, alpha=0.0):
    """Updates a previously fitted metalog object with new data.

    Args:
        m (:obj:`metalog`): The previously fitted metalog object to be updated with `new_data`.
          - `save_data` parameter must have been set equal to True in original metalog fit.

        new_data (:obj:`list` | `numpy.ndarray` | `pandas.Series`): Input data to update the metalog object with.
          - must be an array of allowable types: int, float, numpy.int64, numpy.float64

        penalty (:obj:`str`, optional): Used to specify the norm used in the regularization.
            - must be in set ('l2', None)
                * 'l2' performs Ridge Regression instead of OLS
                    - Automatically shrinks a coefficients, leading to "smoother" fits
            - should be set in conjunction with `alpha` parameter
            - Default: None

        alpha (:obj:`float`, optional): Regularization term to add to OLS fit.
            - strictly >= 0.
            - should be set in conjunction with `penalty` parameter
            - Default: 0. (no regularization, OLS)

    Returns:
        (:obj:`metalog`): Input metalog object that has been updated using `new_data`

    Raises:
      ValueError: 'Input metalog `m.save_data` parameter must be True'
      TypeError: 'Input x must be an array or pandas Series'
      TypeError: 'Input x must be an array of allowable types: int, float, numpy.int64, or numpy.float64'
      IndexError: 'Input x must be of length 3 or greater'
    """

    if not m.save_data:
        raise ValueError("Input metalog `m.save_data` parameter must be True")
    if (
        (type(new_data) != list)
        and (type(new_data) != np.ndarray)
        and (type(new_data) != pd.Series)
    ):
        raise TypeError("Input x must be an array or pandas Series")
    if isinstance(new_data, pd.Series):
        new_data = new_data.values.copy()
    if not all([isinstance(x, (int, float, np.int64, np.float64)) for x in new_data]):
        raise TypeError(
            "Input x must be an array of allowable types: int, float, numpy.int64, or numpy.float64"
        )
    if np.size(new_data) < 3:
        raise IndexError("Input x must be of length 3 or greater")

    old_append_new_data = np.append(m.x, new_data)

    updated_metalog = metalog(
        old_append_new_data,
        bounds=m.output_dict["params"]["bounds"],
        boundedness=m.output_dict["params"]["boundedness"],
        term_limit=m.output_dict["params"]["term_limit"],
        term_lower_bound=m.output_dict["params"]["term_lower_bound"],
        step_len=m.output_dict["params"]["step_len"],
        probs=None,
        fit_method=m.output_dict["params"]["fit_method"],
        penalty=penalty,
        alpha=alpha,
        save_data=True,
    )

    Y = updated_metalog.output_dict["Y"].values
    gamma = Y.T.dot(Y)
    updated_metalog.output_dict["params"]["bayes"]["gamma"] = gamma
    updated_metalog.output_dict["params"]["bayes"]["mu"] = updated_metalog.output_dict[
        "A"
    ]
    v = list()
    for i in range(
        updated_metalog.output_dict["params"]["term_lower_bound"],
        updated_metalog.output_dict["params"]["term_limit"] + 1,
    ):
        v.append(updated_metalog.output_dict["params"]["nobs"] - i)
    v = np.array(v)
    a = v / 2
    updated_metalog.output_dict["params"]["bayes"]["a"] = a
    updated_metalog.output_dict["params"]["bayes"]["v"] = v

    # for now, just using 3 term standard metalog
    v = v[1]
    a = a[1]
    s = np.array([0.1, 0.5, 0.9])
    Ys = np.repeat(1.0, 3)

    Ys = np.column_stack(
        [np.repeat(1, 3), np.log(s / (1 - s)), (s - 0.5) * np.log(s / (1 - s))]
    )
    three_term_metalog_fit_idx = "a{}".format(updated_metalog.term_limit - 3)
    q_bar = np.dot(
        Ys, updated_metalog.output_dict["A"][three_term_metalog_fit_idx].values[-3:]
    )

    updated_metalog.output_dict["params"]["bayes"]["q_bar"] = q_bar

    est = (q_bar[2] - q_bar[1]) / 2 + q_bar[1]
    s2 = ((q_bar[2] - q_bar[1]) / t.ppf(0.9, np.array(v))) ** 2

    gamma = gamma[:3, :3]

    # build covariance matrix for students t
    sig = Ys.dot(np.linalg.solve(gamma, np.eye(len(gamma)))).dot(Ys.T)

    # b = 0.5 * self.output_dict['params']['square_residual_error'][len(self.output_dict['params']['square_residual_error'])]
    b = (a * s2) / gamma[1, 1]
    updated_metalog.output_dict["params"]["bayes"]["sig"] = (b / a) * sig
    updated_metalog.output_dict["params"]["bayes"]["b"] = b

    return updated_metalog
