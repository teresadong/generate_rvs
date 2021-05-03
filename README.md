# Generate Random Variates from Various Distributions


The library for generating random variates is in the file genDistribution.py. By importing the library, we can create an object of the distGenerator class.  

```
# Import the genDistribution Library
import generateDistribution as genDist

# Create an object of the distGenerator class
dist = genDist.distGenerator()
```

Once the object is created, we can call the functions below:

## Generalized Utility Functions

* **plotDist()**: Plot histogram of the generated random numbers

* **compPlotDist()**: Side by Side Comparison of Distribution Histograms, usually the generated Random Variate (from the genDistribution.py Library) and the corresponding scipy function

* **validateDist()**: Validates if the generated distribution matches that of the scipy reference distribution by conducting a Kolmogorov–Smirnov test (KS) Test,   usually the generated Random Variate (from the genDistribution.py Library) and the corresponding scipy function generated Random Variate

## Generate Random Variates for Various Distributions
* **generateExponential(Lambda, size, random_state)**: Generates size number of Exponential Random Variables based on Lambda parameter and random state seed

* **generateWeibull(Alpha, Beta, size, random_state)**: Generates size number of Weibull Random Variables based on Alpha and Beta parameter and random state seed

* **generateBernoulli(p, size, random_state)**: Generates size number of Bernoulli Random Variables based on p parameter and random state seed

* **generateGeometric(p, size, random_state)**: Generates size number of Geometric Random Variables based on p parameter and random state seed

* **generateNormal(Mu, Sigma, size, random_state1, random_state2)**: Generates size number of Normal Random Variables based on Mu, Sigma parameters and random state seeds using the Box-Mueller method

* **generateGamma(Alpha, Beta, size, random_state)**: Generates size number of Gamma Random Variables based on Alpha and Beta parameter and random state seed

More exmaples of how to use the functions are in demo.ipynb