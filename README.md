<h1 align="center"></h1>

<h1 align="center">&iota;&zeta;</h1>

<p align="center">&iota;&zeta; (romanized to: iz) is Common Ionian for 16.</p>
<p align="center"> It was in the year 2016 that Thomas Klien Published His <a href="http://www.metalogdistributions.com/images/The_Metalog_Distributions_-_Keelin_2016.pdf">Metalog Distributions</a> Paper</p>


---


Probability distributions are not something that tends to make headlines. However, in 2016, a new distribution was introduced that has the potential to upgrade the commonly used distributions to 
a toolset that is less fussy and more flexible than the standard families. The metalog distribution is a quantile parameterized distribution. It is an alternative for something is information sparse as a triangular distribution. It can also fit large datasets in a more flexible way than the standard families.

---

## Specification

$$
y=Q(x)=\mu+s\ln\Bigl({x\over{1-x}}\Bigr)
$$

$$
\mu =\beta_1 + \beta_4(x -0.5) + \beta_5(x -0.5)^2 + \beta_7(x -0.5)^3 + \beta_9(x -0.5)^4 + \dots
$$

$$
s = \beta_2 + \beta_3(x -0.5) + \beta_6(x -0.5)^2 + \beta_8(x -0.5)^3 + \beta_{10}(x -0.5)^4 + \dots
$$

Each $(y_i,x_i)$ pair represents quantile pair. For example y is the value at evaluated at some x quantile.

### Fitting the SPT (symmetric-percentile triplet)

A SPT pair is the $(y_i,x_i)$ set when evaluating data on the $x = (.10,.50,.90)$ quantiles. In some applications this SPT metalog can serve as an alternative to the triangular distribution. 

Using the equations from the Specification section we can fit the SPT metalog with the following equations.

$$
\mathbf{y} =
 \begin{pmatrix}
  y_{1}\\
  y_{2}\\
  y_{3}
 \end{pmatrix},
\mathbf{x} =
 \begin{pmatrix}
  1 & ln(\frac{x_1}{1 - x_1}) & (x_1 - 0.5)ln(\frac{x_1}{1 - x_1})  \\
  1 & ln(\frac{x_2}{1 - x_2}) & (x_2 - 0.5)ln(\frac{x_2}{1 - x_2})  \\
  1 & ln(\frac{x_3}{1 - x_3}) & (x_3 - 0.5)ln(\frac{x_3}{1 - x_3})  \\
 \end{pmatrix},
\mathbf{\beta} =
 \begin{pmatrix}
  \beta_{1}\\
  \beta_{2}\\
  \beta_{3}
 \end{pmatrix}
$$

## Installation

### 1. Clone
   
a. HTTPS: 

```sh
git clone https://github.com/alexhallam/iz.git
```

or

b. GitHub CLI: 

```sh 
gh repo clone alexhallam/iz
```

### 2. Change Directory

```sh
cd iz
```

### 3. Installation Package

```sh
# --break-system-packages is a flag that was needed in the latest Debian 12 release
pip install dist/*.tar.gz --break-system-packages
```

## Examples

```py
# install
gh repo clone alexhallam/iz`
cd iz
poetry install
```

### Fish Size Example

This data may be originally sourced from the `pymetalog` and/or `rmetalog` projects. Attribution goes to the original authors.

```py
import pandas as pd
import iz
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/alexhallam/iz/main/iz/data/fishSize.csv")
fish_metalog = iz.metalog(y=df.FishSize, bounds=[0,60], boundedness='b', terms=3, step_len=.001, epochs=500, lr = 0.1)
iz.summary(fish_metalog)

r_gens = iz.rmetalog(fish_metalog, n=5000, generator="hdr")
plt.hist(r_gens, 12)
plt.show()

# quantiles from a percentile
qs = iz.qmetalog(fish_metalog, y=[0.25, 0.5, 0.999])
print("qmetalog demo: " + str(qs))

# probabilities from a quantile
ps = iz.pmetalog(fish_metalog, q=[3, 10, 35])
print("pmetalog demo: " + str(ps))

# density from a quantile
ds = iz.dmetalog(fish_metalog, q=[3, 10, 25])
print("dmetalog demo: " + str(ds))
```

### Bounding

There are 4 bounding options. The bounding options are used to restrict the domain of the metalog distribution. The bounding options are:

1. unbounded `u`
2. semi-lower bounded `sl`
3. semi-upper bounded `su`
4. fully bounded `b`


The $y$ values are transformed for each of these cases as follows:

1. unbounded `u` : $y = x$
2. semi-lower bounded `sl` : $y = \ln(y - b_l)$
3. semi-upper bounded `su` : $y = -\ln(b_u - y)$
4. bounded `b` : $y = \ln(\frac{x - b_l}{b_u - x})$


These bounds increase the flexibility of the distribution family.

## References

[Metalog Distributions](http://www.metalogdistributions.com/images/The_Metalog_Distributions_-_Keelin_2016.pdf) Thomas Klein, 2016

### Inspiration

[pymetalog](https://github.com/tjefferies/pymetalog) - Python implementation of the metalog distribution. `iz` is a fork of this project. So, saying it is simply an 'inspiration' is underselling it. I simply changed the underlying computation to pytorch and I use stochastic gradient descent to fit the parameters as opposed to the OLS and Linear Programming methods used in the original implementation.

[rmetalog](https://github.com/isaacfab/rmetalog) - R implementation of the metalog distribution.



