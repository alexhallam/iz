import polars as pl
import torch

def summary(m):
    """Prints information about the fitted metalog m.
        Prints to console:

    Args:
        m (:obj:`metalog`): A fitted metalog object.

    """
    #for each key in m.output_dict['params'], print the key and the value
    # Set print options for tensors
    y = (m.output_dict['model_data']['y'].numpy()).flatten()
    df_model_data = pl.DataFrame({'y': y})
    X = (m.output_dict['model_data']['X'].numpy()).flatten()
    # for each column in m.output_dict['data_model']['X'], append a new column to df_model_data in a dataframe and flatten each column
    X = m.output_dict['model_data']['X'].numpy()
    for col_idx in range(X.shape[1]):
        col_name = f'X_{col_idx}'
        df_model_data = df_model_data.with_columns(pl.Series(col_name, X[:, col_idx]))

    df_coeffs = pl.DataFrame({'coeffs': m.output_dict['model_data']['coeffs'].numpy().flatten()})
    
    bounds = [x if (type(x) != torch.Tensor) else x.numpy().astype(int) for x in m.output_dict["params"]["bounds"]]

    print(
    " -----------------------------------------------\n",
    "Summary of Metalog Distribution Object\n",
    "-----------------------------------------------\n",
    "\nParameters",
    "\n-----------------------------------------------",
    "\nBounds: ",
    bounds,
    "\nBoundedness: ",
    m.output_dict["params"]["boundedness"],
    "\nTerms: ",
    m.output_dict["params"]["terms"],
    "\nStep Length: ",
    m.output_dict["params"]["step_len"],
    "\nNumber of Observations: ",
    m.output_dict["params"]["nobs"],
    "\nLearning Rate: ",
    m.output_dict["params"]["lr"],
    "\nEpochs: ",
    m.output_dict["params"]["epochs"],
    "\nWeight Decay: ",
    m.output_dict["params"]["weight_decay"],
    "\n\nModel Data",
    "\n-----------------------------------------------\n",
    df_model_data,
    "\n\nCoefficients",
    "\n-----------------------------------------------\n",
    df_coeffs,
    )
