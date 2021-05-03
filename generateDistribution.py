# Import pandas
import pandas as pd
# Import numpy
import  numpy as np
import math
# Import maptplotlib
import matplotlib.pyplot as plt

#Import stats from scipy to validate distributions
from scipy import stats

class distGenerator:
    def __init__(self):
        pass
    
    def plotDist(self,dist_name,random_numbers):
        """Plot histogram of the generated random numbers

        Args:
            dist_name (str): Name of the Distribution (with Parameters, if applicable)
            random_numbers (list<float>): Array of Random Variables
        """
        
        plt.title(f"Random Variates for {dist_name} Distribution")
        plt.hist(random_numbers,bins='auto')
        plt.xlabel('Random Number', fontweight ='bold')
        plt.ylabel('Frequency', fontweight ='bold')
        plt.show()
        
    
    def compPlotDist(self,dist_name,rvs_ud,rvs_ref):
        """Side by Side Comparison of Distribution Histograms

        Args:
            dist_name (str): Name of the Distribution (with Parameters, if applicable)
            rvs_ud (list<float>): Array of Random Variables, usually library generated
            rvs_ref (list<float>): Array of Random Variables, usually SciPy reference
        """

        fig, (ax1, ax2) = plt.subplots(1,2)
        
        fig.suptitle(f"Comparative Plot for {dist_name} Distribution")
        ax1.hist(rvs_ud,bins='auto')
        ax1.set_title('Library Generated Dist')
        ax1.set(xlabel ='Random Number', ylabel='Frequency')
        ax1.label_outer()
        
        ax2.hist(rvs_ref,bins='auto')
        ax2.set_title('Scipy Reference Dist')
        ax2.set(xlabel ='Random Number', ylabel='Frequency')
        ax2.label_outer()
                
        plt.show()
        

    def validateDist(self,dist_name,rvs_ud,rvs_ref,alpha=0.05):
        """Validates if the generated distribution matches that of the scipy reference distribution by conducting a Kolmogorov–Smirnov test (KS) Test

        Args:
            dist_name (str): Name of the Distribution (with Parameters, if applicable)
            rvs_ud (list<float>): Array of Random Variables, usually library generated
            rvs_ref (list<float>): Array of Random Variables, usually SciPy reference
            alpha (float, optional): P-Value Threshold for the KS Test. Defaults to 0.05.
        """
        
        
        ks_test_result =  stats.kstest(rvs_ud,rvs_ref)
        print('Result of Kolmogorov–Smirnov test (KS) Test: ')
        print('HO: Distributions are Identical')
        print('HA: Distributions are Different')
        print(f"P-value is {ks_test_result.pvalue}")
        
        if ks_test_result.pvalue < alpha:
            print(f"Not a {dist_name} Distribution since null hypothesis is rejected")
        else:
            print(f"{dist_name} Distribution since null hypothesis is NOT rejected")
            
    
    #--------------------- Generate Random Variates ----------------#
    def generateExponential(self, Lambda, size, random_state):
        """Generates size number of Exponential Random Variables based on Lambda parameter and random seed seed

        Args:
            Lambda (float): Rate parameter (lambda) for exponential distribution
            size (int): Number of random variates to generate
            random_state (int): Random seed for the exponential variable

        Returns:
            list<float>: List of Random Variates from the Exponential Distribution
        """  

        U = stats.uniform.rvs(size=size,random_state=random_state)
        
        R = -(1/Lambda) * (np.log(1 - U))
        
        return R


    def generateWeibull(self, Alpha, Beta, size, random_state):
        """Generates size number of Weibull Random Variables based on Alpha and Beta parameter and random state seed

        Args:
            Alpha (float): exponentiation parameter, where alpha=1  is the non-exponentiated Weibull distribution
            Beta (float): shape parameter for the non-exponentiated Weibull law
            size (int): Number of random variates to generate
            random_state (int): Random seed for the weibull variable

        Returns:
            list<float>: List of Random Variates from the Weibull(Alpha, Beta) Distribution
        """

        U = stats.uniform.rvs(size=size,random_state=random_state)        
        
        R = pow((-1 * np.log (1 - U)),(1/Beta))/Alpha

        return R


    def generateBernoulli(self, p, size, random_state):
        """Generates size number of Bernoulli Random Variables based on p parameter and random state seed

        Args:
            p (float): p is probability of single success, 1-p is probability of single failure
            size (int): Number of random variates to generate
            random_state (int): Random seed for the bernoulli variable

        Returns:
            list<float>: List of Random Variates from the Bernoulli(p) Distribution
        """
        
        U = stats.uniform.rvs(size=size,random_state=random_state)
        
        B = (U<=p).astype(int)
        
        return B


    def generateGeometric(self, p, size, random_state):
        """Generates size number of Geometric Random Variables based on p parameter and random state seed

        Args:
            p (float): p is probability of single success, 1-p is probability of single failure
            size (int): Number of random variates to generate
            random_state (int): Random seed for the geometric variable

        Returns:
            list<float>: List of Random Variates from the Geometric(p) Distribution
        """
        
        U = stats.uniform.rvs(size=size,random_state=random_state)
        
        R = np.log(1-U)/np.log(1-p)
        
        R_discrete = []
        for r in R:
            R_discrete.append(math.ceil(r))

        return R_discrete
            
    
    def multiplyList(self,myList) :
        # Multiply elements one by one
        result = 1
        for x in myList:
            result = result * x
        return result
    
    def generateNormal(self, Mu, Sigma, size, random_state1, random_state2):
        """Generates size number of Normal Random Variables based on Mu, Sigma parameters and random state seeds using the Box-Mueller method

        Args:
            Mu (float): mean of the normal distribution
            Sigma (float): standard deviation of the normal distribution
            size ([type]): [description]
            random_state1 (int): Random seed for the normal variable
            random_state2 (int): Random seed for the normal variable

        Returns:
            list<float>: List of Random Variates from the Normal(Mu, Sigma) Distribution
        """
        
        U1 = stats.uniform.rvs(size=size,random_state=random_state1)        
        U2 = stats.uniform.rvs(size=size,random_state=random_state2)       
        
        # Standard Normal pair
        Z0 = np.sqrt(-2*np.log(U1))*np.cos(2*np.pi*U2)
        Z1 = np.sqrt(-2*np.log(U1))*np.sin(2*np.pi*U2)
            
        # Scaling                        
        Z2 = Z0*Sigma+Mu
            
        return Z2
            
    def generateGamma(self, Alpha, Beta, size, random_state):
        """Generates size number of Gamma Random Variables based on Alpha and Beta parameter and random state seed

        Args:
            Alpha (float): shape parameter for alpha, when its an integer, gamma becomes an erlang distribution, when its 1 gamma becomes the exponential distribution
            Beta (float): parameter for shifting the distribution
            size (int): Number of random variates to generate
            random_state (int): Random seed for the gamma variable

        Returns:
            list<float>: List of Random Variates from the Gamma(Alpha, Beta) Distribution
        """
        
        
        G = np.zeros(size)
        
        for i in range(1,size):
            U = stats.uniform.rvs(size=Alpha, random_state=random_state+i) 

            # Based on Formula: https://arxiv.org/pdf/1304.3800.pdf
            G[i] = -1/Beta*np.log(self.multiplyList(U))


        return G
            
