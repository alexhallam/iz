import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

def get_probability_grid(y_old, step_len):
    """Computes a sequence of probabilities given the step length and the input data.
    
    Args:
        y_old (:obj:`list` | `numpy.ndarray` | `pandas.Series` | `torch.tensor`): Input data to fit a metalog to.
        step_len (:obj:`float`, optional): Used to specify the bin width used to estimate the metalog.
        
    Returns:
        y (:obj:`dict`): Dictionary with keys 'y' and 'probs' that are torch tensors.
            - y['y']: (:obj:`torch.tensor`): Sorted input data.
            - y['probs']: (:obj:`torch.tensor`): Probabilities associated with the data values in y['y'].
            
    """

    # Convert y_old to a PyTorch tensor
    y_old_tensor = torch.tensor(y_old.clone().detach(), dtype=torch.float32)

    # Sort y_old_tensor and create x tensor
    y_sorted, _ = torch.sort(y_old_tensor)
    y = torch.zeros((len(y_old), 2), dtype=torch.float32)
    y[:, 0] = y_sorted

    # Compute probabilities
    l = len(y_sorted)
    y[:, 1] = 0
    for i in range(l):
        if i == 0:
            y[i, 1] = 0.5 / l
        else:
            y[i, 1] = y[i - 1, 1] + 1 / l

    # If the number of data points is greater than 100, then use a step length to reduce the number of probabilities.
    if len(y) > 100:
        y2 = torch.linspace(step_len, 1 - step_len, int((1 - step_len) / step_len))

        tailstep = step_len / 10

        y1 = torch.linspace(
            tailstep, (y2.min() - tailstep), int((y2.min() - tailstep) / tailstep) + 1
        )

        y3 = torch.linspace(
            (y2.max() + tailstep),
            (y2.max() + tailstep * 9),
            int((tailstep * 9) / tailstep) ,
        )

        probs = torch.cat((y1, y2, y3))
        y_new = torch.quantile(y_old_tensor, probs)
        dict_y = {"x": y_new, "probs": probs}
        y = dict_y

    return y

class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # One input feature, one output
        # it is necessary to convert the weight and bias to float64 to avoid errors during backpropagation
        self.linear.weight = nn.Parameter(self.linear.weight.to(torch.float64))  # Convert weight to float64
        self.linear.bias = nn.Parameter(self.linear.bias.to(torch.float64))  # Convert bias to float64

    def forward(self, x):
        return self.linear(x)

epoch_data = {}
m_dict = {}
def a_vector_SGD(X, y, learning_rate=0.1, num_epochs=5000, weight_decay=0.0, convergence_threshold=1e-15, debug=False):
    input_dim = X.shape[1] # Number of columns in X
    model = LinearRegression(input_dim) # Instantiate the model
    criterion = nn.MSELoss() # Define the loss function
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    prev_loss = None
    for epoch in range(num_epochs):
        outputs = model(X.to(torch.float64))
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            coeffs_i = torch.cat((model.linear.bias.data, model.linear.weight.data.view(-1)))
            epoch_data[epoch] = {
                'epoch': epoch + 1,
                'loss': f'{loss.item():.4f}',
                }
            for i, coeff in enumerate(coeffs_i):
                epoch_data[epoch][f'coeff_{i+1}'] = f'{coeff.item():.4f}'
            print(f'epoch_data: {epoch_data[epoch]}')
        if epoch >= 1 and prev_loss is not None:
            if abs(loss.item() - prev_loss) < convergence_threshold:
                print(f"Convergence of {convergence_threshold} reached at epoch {epoch + 1}")
                break
        prev_loss = loss.item()
    coeffs = torch.cat((model.linear.bias.data, model.linear.weight.data.view(-1)))
    if debug:
        print(f'coeffs: {coeffs}')
        
    m_dict["y"] = y
    m_dict["X"] = X
    m_dict["coeffs"] = coeffs
    
    return m_dict

class metalog:
    def __init__(
    self,
    y,
    bounds=[0, 1],
    boundedness="u",
    step_len=0.01,
    probs=None,
    terms = 3,
    lr = 0.1,           # learning rate
    epochs = 5000,
    weight_decay = 0.0 # L2 regularization
    ):
        self.y = y
        self.boundedness = boundedness
        self.bounds = bounds[:]
        self.terms = terms
        self.step_len = step_len
        self.probs = probs
        self.nobs = len(y)
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        dict_x = {}
        self.output_dict = {} 
        dict_x["x"] = self.y
        if probs == None:
            dict_x = get_probability_grid(self.y, step_len=step_len)
        else:
            dict_x["probs"] = self.probs # torch.tensor(self.probs, dtype=torch.float64)
        output_dict = {}

        # build z vector based on boundedness
        dict_x = self.append_zvector(dict_x)
        self.output_dict["params"] = self.get_params()
        output_dict["dataValues"] = dict_x

        # Construct the Y Matrix initial values
        Y = {}
        Y["y1"] = torch.ones(len(dict_x["x"])).reshape(-1, 1)
        Y["y2"] = torch.log_(dict_x["probs"] / (1 - dict_x["probs"]))
        Y["y3"] = (dict_x["probs"] - 0.5) * Y["y2"]

        if self.terms > 3:
            Y["y4"] = dict_x["probs"] - 0.5

        # Complete the values through the term limit
        if terms > 4:
            for i in range(5, self.terms + 1):
                yn = "y" + str(i)

                if i % 2 != 0:
                    Y[yn] = Y["y4"] ** (int(i // 2))

                if i % 2 == 0:
                    zn = "y" + str(i - 1)
                    Y[yn] = Y["y2"] * Y[zn]

        output_dict["Y"] = Y
        # Reshape each tensor in output_dict["Y"]["yi"] to be a 2d tensor if it is 1D. if it is already 2D, then do nothing.
        for key, value in output_dict["Y"].items():
            if len(value.shape) == 1:
                output_dict["Y"][key] = value.reshape(-1, 1)
            else:
                output_dict["Y"][key] = value
    
        output_dict["Y_tensor_ones"] = torch.stack(list(Y.values()), dim=1).reshape(len(output_dict["Y"]['y1']), len(output_dict["Y"])) # remove the third dimension to get a 2D tensor
        output_dict["Y_tensor"] = (output_dict["Y_tensor_ones"][:, 1:]).to(torch.float64)
        output_dict["z"] = dict_x['z'].reshape(-1, 1).to(torch.float64)
        self.output_dict['model_data'] = a_vector_SGD(
            X = output_dict["Y_tensor"],
            y = dict_x['z'].reshape(-1, 1).to(torch.float64), # z is the same as the target vector in the unbounded case (i.e. boundedness = 'u')
            learning_rate=lr, 
            weight_decay=weight_decay,
            num_epochs=epochs, 
            convergence_threshold=1e-30,
            debug=True
        )
                
        
    # input validation functions for metalog

    @property
    def y(self):
        """y (:obj:`list` | `numpy.ndarray` | `pandas.Series` | `tensor.torch`): Input data to fit a metalog to."""
        # This decorator defines a getter method for the property y. The getter allows you to access the value of y as if it were an attribute of an instance of this class. For eyample, if you have an instance of this class called metalog, you can access the value of y by typing metalog.y. The getter method simply returns the value stored in the private attribute _y.

        # The getter method simply returns the value stored in the private attribute _y.

        return self._y

    @y.setter
    def y(self, ys):
        if (type(ys) != list) and (type(ys) != np.ndarray) and (type(ys) != pd.Series) and (type(ys) != torch.Tensor):
            raise TypeError("Input y must be an array or pandas Series or torch tensor")
        if isinstance(ys, pd.Series):
            ys = ys.values.copy() # convert to numpy array
            ys = torch.tensor(ys, dtype=torch.float64) # convert numpy array to torch tensor
        if isinstance(ys, np.ndarray):
            ys = torch.tensor(ys, dtype=torch.float64)
        if isinstance(ys, list):
            ys = torch.tensor(ys, dtype=torch.float64)
        if torch.numel(ys) < 3:
            raise IndexError("Input y must be of length 3 or greater")
        self._y = ys
        
    @property
    def bounds(self):
        """bounds (:obj:`list`, optional): Upper and lower limits to filter the data with before calculating metalog quantiles/pdfs."""
        # This decorator defines a getter method for the property bounds. The getter allows you to access the value of bounds as if it were an attribute of an instance of this class. For example, if you have an instance of this class called metalog, you can access the value of bounds by typing metalog.bounds. The getter method simply returns the value stored in the private attribute _bounds.
        return self._bounds
    
    
    @bounds.setter
    def bounds(self, bs):
        if type(bs) != list:
            raise TypeError("bounds parameter must be of type list")
        if not all(isinstance(x, (int)) for x in bs):
            raise TypeError("bounds parameter must be list of integers")
        if (self.boundedness == "sl" or self.boundedness == "su") and len(bs) != 1:
            raise IndexError(
                "Must supply only one bound for semi-lower or semi-upper boundedness"
            )
        if self.boundedness == "b" and len(bs) != 2:
            raise IndexError(
                "Must supply exactly two bounds for bounded boundedness (i.e. [0,30])"
            )
        if self.boundedness == "su":
            bs_o = [torch.min(self.y), bs[0]]
        if self.boundedness == "sl":
            bs_o = [bs[0], torch.max(self.y)]
        if self.boundedness == "b" or self.boundedness == "u":
            bs_o = bs
        if self.boundedness == "sl" and torch.min(self.y) < bs_o[0]:
            raise ValueError(
                "for semi-lower boundedness the lower bound must be less than the smallest value in x"
            )
        if self.boundedness == "su" and torch.max(self.y) > bs_o[1]:
            raise ValueError(
                "for semi-upper boundedness the upper bound must be greater than the largest value in x"
            )
        #if bs_o[0] > bs_o[1] and self.boundedness == "b":
        #    raise ValueError("Upper bound must be greater than lower bound")
        self._bounds = bs_o    
        
        
    @property
    def boundedness(self):
        """boundedness (:obj:`str`, optional): String that is used to specify the type of metalog to fit."""
        # This decorator defines a getter method for the property boundedness. The getter allows you to access the value of boundedness as if it were an attribute of an instance of this class. For example, if you have an instance of this class called metalog, you can access the value of boundedness by typing metalog.boundedness. The getter method simply returns the value stored in the private attribute _boundedness.
        return self._boundedness

    @boundedness.setter
    def boundedness(self, bns):
        if bns != "u" and bns != "b" and bns != "su" and bns != "sl":
            raise ValueError("boundedness parameter must be u, su, sl or b only")
        self._boundedness = bns
        
    @property
    def terms(self):
        """terms (:obj:`int`, optional): The number of coefficients to generate."""

        return self._terms

    @terms.setter
    def terms(self, tl):
        if type(tl) != int:
            raise TypeError(
                "term_limit parameter should be an integer between 3 and 30"
            )
        if tl > 30 or tl < 3:
            raise ValueError(
                "term_limit parameter should be an integer between 3 and 30"
            )
        if tl > len(self.y):
            raise ValueError(
                "term_limit must be less than or equal to the length of the vector x"
            )
        self._terms = tl
        
    @property
    def step_len(self):
        """step_len (:obj:`float`, optional): Used to specify the bin width used to estimate the metalog."""

        return self._step_len

    @step_len.setter
    def step_len(self, sl):
        if sl < 0.001 or sl > 0.01:
            raise ValueError("step_len must be >= to 0.001 and <= to 0.01")
        self._step_len = sl
        
    @property
    def probs(self):
        """probs (:obj:`list` | `numpy.ndarray` | `torch.tensor` optional): Probabilities associated with the data values in x."""

        return self._probs

    @probs.setter
    def probs(self, ps):
        def check_tensor_types(tensor):
            dtype = tensor.dtype
            if dtype == torch.int32 or dtype == torch.int64 or dtype == torch.float32 or dtype == torch.float64:
                return True
            else:
                return False
        if ps is not None:
            if not isinstance(ps, (list, np.ndarray, torch.Tensor)):
                raise TypeError("Input probabilities must be a list, array, or torch tensor")
            if isinstance(ps, (list, np.ndarray)):    
                if not all(isinstance(x, (int, float)) for x in ps):
                    raise TypeError(
                        "Input probabilities must be an array of integer or float data"
                    )
                if np.max(ps) > 1 or np.min(ps) < 0:
                    raise ValueError(
                        "Input probabilities must have values between, not including, 0 and 1"
                    )
                if np.size(np.where(np.isnan(ps))) != 0:
                    raise ValueError("Input probabilities cannot contain nans")                    
            if isinstance(ps, torch.Tensor):
                if not check_tensor_types(ps):
                    raise TypeError(
                        "Input probabilities must be an array of integer or float data"
                    )
                if torch.max(ps) > 1 or torch.min(ps) < 0:
                    raise ValueError(
                        "Input probabilities must have values between, not including, 0 and 1"
                    )
                # torch Input probabilities cannot contain nans
                nan_indices = torch.isnan(ps)
                if torch.any(nan_indices):  # Check if any NaN values exist
                    raise ValueError("Input probabilities cannot contain nans")

            if len(ps) != len(self.y):
                raise IndexError("probs vector and x vector must be the same length")
            if isinstance(ps, (list, np.ndarray)):   
                ps = ps.copy()
            if isinstance(ps, torch.Tensor):
                ps = ps.clone()
        self._probs = ps
        
        @property
        def nobs(self):
            """nobs (:obj:`int`): Number of observations in the input data."""
            return self._nobs
        
        @nobs.setter
        def nobs(self, n):
            if type(n) != int:
                raise TypeError("nobs parameter must be an integer")
            if n < 3:
                raise ValueError("nobs parameter must be greater than 2")
            self._nobs = n
        
        @property
        def lr(self):
            """lr (:obj:`float`, optional): Speed"""
            return self._lr
        
        @lr.setter
        def lr(self, lr):
            if lr < 0.0001 or lr > 0.1:
                raise ValueError("lr must be >= to 0.0001 and <= to 0.1")
            self._lr = lr
        
        @property
        def epochs(self):
            """epochs (:obj:`int`, optional): Number of iterations used for backprop."""
            return self._epochs
        
        @epochs.setter
        def epochs(self, epochs):
            if epochs < 1 or epochs > 1000000:
                raise ValueError("epochs must be >= to 1 and <= to 1000000")
            self._epochs = epochs
            
        @property
        def weight_decay(self):
            """weight_decay (:obj:`float`, optional): The weight decay is a way to penalize the fit via L2 regularization."""
            return self._weight_decay
        
        @weight_decay.setter
        def weight_decay(self, weight_decay):
            if weight_decay < 0.0 or weight_decay > 1.0:
                raise ValueError("weight_decay must be >= to 0.0 and <= to 1.0")
            self._weight_decay = weight_decay
    
# functions for metalog

    def append_zvector(self, dict_x):
        """The zvector is what becomes the new y after applying the transformations for the different boundedness types.
            Sets the `dataValues` key (Dictionary of torch tensors) in `output_dict` to a DataFrame with columns ['x','probs','z'] of type numeric.

        Uses `boundedness` attribute to set z vector
            - 'u': output_dict['dataValues']['z'] = x
                * Start with all the input data
            - 'sl': output_dict['dataValues']['z'] = log( (x-lower_bound) )
            - 'su': output_dict['dataValues']['z'] = log( (upper_bound-x) )
            - 'b': output_dict['dataValues']['z'] = log( (x-lower_bound) / (upper_bound-x) )

        Returns:
            df_x: (:obj:`pandas.DataFrame` with columns ['x','probs','z'] of type numeric): DataFrame that is used as input to `a_vector_OLS_and_LP` method.
                - df_x['x']: metalog.x
                - df_x['probs']: metalog.probs
                - df_x['z']: z vector above
        """

        if self.boundedness == "u":
            dict_x["z"] = dict_x["x"]
        if self.boundedness == "sl":
            dict_x["z"] = torch.log((dict_x["x"] - torch.tensor(self.bounds[0], dtype=torch.float64)).clone().detach())
        if self.boundedness == "su":
            dict_x["z"] = -torch.log(torch.tensor((torch.tensor(self.bounds[1], dtype=torch.float64).clone().detach() - dict_x["x"]), dtype=torch.float64))
        if self.boundedness == "b":
            dict_x["z"] = torch.log(
                torch.tensor(
                    ((dict_x["x"] - torch.tensor(self.bounds[0], dtype=torch.float64).clone().detach()) / (torch.tensor(self.bounds[1], dtype=torch.float64).clone().detach() - dict_x["x"])),
                    dtype=torch.float64,
                )
            )

        return dict_x
    
    def get_params(self):
        """Sets the `params` key (dict) of `output_dict` object prior to input to `a_vector_OLS_and_LP` method.
            - Uses metalog attributes to set keys

        Returns:
            params: (:obj:`dict`): Dictionary that is used as input to `a_vector_OLS_and_LP` method.

        """

        params = {}
        params["bounds"] = self.bounds
        params["boundedness"] = self.boundedness
        params["terms"] = self.terms
        params["step_len"] = self.step_len
        params["nobs"] = self.nobs
        params["lr"] = self.lr
        params["epochs"] = self.epochs
        params["weight_decay"] = self.weight_decay
        

        return params
    
    