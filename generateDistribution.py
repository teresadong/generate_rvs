# Import pandas
import pandas as pd
# Import numpy
import  numpy as np

# Import maptplotlib
import matplotlib.pyplot as plt

class distGenerator:
    def __init__(self):
        pass
    
    def plotDist(self,dist_name,random_numbers):
        # Plot histogram of the generated random numbers
        plt.title(f"Random Variates for {dist_name} Distribution")
        plt.hist(random_numbers,bins='auto')
        plt.xlabel('Random Number', fontweight ='bold')
        plt.ylabel('Frequency', fontweight ='bold')
        plt.show()

    def generateUniformLCG(self,mult=16807,mod=(2**31)-1,seed=123,size=1):          
        """A reasonably good pseudo random generator

        Args:
            mult (int, optional): multiplier. Defaults to 16807.
            mod ([type], optional): modulus. Defaults to (2**31)-1.
            seed (int, optional): seed. Defaults to 123.
            size (int, optional): number of numbers to generate. Defaults to 1.

        Returns:
            [type]: [description]
        """
        U = np.zeros(size)
        
        x = (seed*mult+1)%mod
        
        U[0]= x/mod
        
        # Generating n random numbers
        for i in range(1,size):
            x = (x*mult+1)%mod
            U[i] = x/mod
        
        return U
    
    def generateUniform(self,low=0,high=1,seed=123,size=1):
        """Generate unformly random number between `low` and `high` limits

        Args:
            low (int, optional): [description]. Defaults to 0.
            high (int, optional): [description]. Defaults to 1.
            seed (int, optional): [description]. Defaults to 123.
            size (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: [description]
        """
        U = self.generateUniformLCG(seed=seed,size=size)
        return low+(high-low)*U
    
    def generateExponential(self,Lambda=1,n=10000,seed=123,size=1):
        
        U = self.generateUniform(seed=seed,size=size)    
        
        R = -(1/Lambda) * (np.log(1 - U))
        
        return R


    def generateWeibull(self,Lambda=1,beta=2,n=10000,seed=123,size=1):
        
        U = self.generateUniform(seed=seed,size=size) 
        
        R = pow((-1 * np.log (1 - U)),(1/beta))/Lambda

        return R

    
    def generateBernoulli(self,p=0.5,seed=123,size=1):
        
        U = self.generateUniform(seed=seed,size=size) 
        
        B = (U<=p).astype(int)
        
        return B




    def generateGeometric(self,p=0.5,seed=123,size=1):
        U = self.generateUniform(seed=seed,size=size)        
        
        R = np.log(1-U)/np.log(1-p)

        return R
            
    
    def multiplyList(self,myList) :
        # Multiply elements one by one
        result = 1
        for x in myList:
            result = result * x
        return result
    
    def generateNormal(self,mu=0.0,sigma=1.0,seed1=123,seed2=124,size=1):

        U1 = self.generateUniform(seed=seed1,size=size)
        U2 = self.generateUniform(seed=seed2,size=size)        
        
        # Standard Normal pair
        Z0 = np.sqrt(-2*np.log(U1))*np.cos(2*np.pi*U2)
        Z1 = np.sqrt(-2*np.log(U2))*np.sin(2*np.pi*U2)
            
        # Scaling                        
        Z2 = Z0*sigma+mu
            
        return Z2
            
    def generateGamma(self,Lambda=1,alpha=2,beta=3,seed=123,size=1):
        
        G = np.zeros(size)
        
        for i in range(1,size):
            U = self.generateUniform(seed=seed+i,size=alpha) 

            # Based on Formula: https://arxiv.org/pdf/1304.3800.pdf
            G[i] = -1/beta*np.log(self.multiplyList(U))


        return G
            
