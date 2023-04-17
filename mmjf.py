import numpy as np
import pandas as pd
from FactorMatrix import factor_matrix
from preprocess import *
from utilities import *
from optimization import *

# this file implemented two classes for storing CMTF process and results.




# basic joint factorization of heterogeneous omics data,using alternating least squares
# for optimization, best for efficiency
''' parameters:
    mfpt: the microbiome functional profiling tensor,sample should have been in the same 
order as the metabolome matrix
    mbm: the metabolome matrix,sample should have been in the same order as the microbiome functional 
profiling tensor
    n_lf: the number of latent factors(components), default is 3
    n_iter: maximum number of iterations,default is 2000.
    tol: the convergence threshold,default is 1e-6
'''
class BasicFac(factor_matrix):
    def __init__(self,
                 mfpt,
                 mbm,
                 n_lf=3,
                 n_iter=2000):
        super().__init__()
        self.tensor = mfpt
        self.matrix = mbm
        self.lf = n_lf
        self.iter = n_iter
        self.mask = None
    

    # factorization method for basic CMTF model
    def factorization(self):
        # data format checking
        try:
            if not isinstance(self.tensor,np.ndarray) or not isinstance(self.matrix,np.ndarray):
                raise TypeError("Invalid tensor type or matrix type")

            if len(self.tensor.shape) != 3:
                raise ValueError("only accept 3-way tensor as input")
            if len(self.matrix.shape) != 2:
                raise ValueError("only accept 2-way matrix as input")


        except ValueError as ve:
            print("error:",ve)
        except TypeError as te:
            print("error:",te)

        if not self.mask:
            res = joint_mt(self.tensor,self.matrix,self.lf)
        else:
            tmask = self.mask[0]
            mmask = self.mask[1]
            res = joint_mt(self.tensor,self.matrix,self.lf,tmask,mmask)
        self.tfm = res.tfm
        self.mfm = res.mfm















#structure revealing joint factorization of heterogeneous omics data,using nonlinear conjugate gradient
# method for optimization , its better for discovering common variation and specific variation between omics datasets
''' parameters:
    mfpt: the microbiome functional profiling tensor,sample should have been in the same 
order as the metabolome matrix
    mbm: the metabolome matrix,sample should have been in the same order as the microbiome functional 
profiling tensor
    n_lf: the number of latent factors(components), default is 3
    n_iter: maximum number of iterations,default is unlimited.
    tol: the convergence threshold,default is 1e-8

'''
class StructRevealFac(factor_matrix):
    def __init__(self,
                 mfpt,
                 mbm,
                 n_lf=3,
                 n_iter=None,
                 tol=1e-8,
                 **kwargs):
        super().__init__()
        argnum = len(kwargs.keys())
        try:
            ALPHA = 1
            BETA_TEN = 1e-3
            BETA_MAT = 1e-3
            EPS = 1e-8
            add_arg = {'alpha':ALPHA,
                       'beta_ten':BETA_TEN,
                       'beta_mat':BETA_MAT,
                       'eps':EPS,
                       }

            if argnum:
                if not set(kwargs.keys()).issubset(set(add_arg.keys())):
                    raise ValueError("unsupported parameter detected")
                for keyparam in add_arg.keys():
                    if keyparam in kwargs.keys():
                        add_arg[keyparam] = kwargs[keyparam]
        except ValueError as ve:
            print('Error:',ve)
        
        self.tensor = mfpt
        self.matrix = mbm
        self.lf = n_lf
        self.iter = n_iter
        self.mask = None
        self.cparam = add_arg
        

    
    #factorization for structure-revealing all in once model
    def factorization(self):
        # data format checking
        try:
            if not isinstance(self.tensor,np.ndarray) or not isinstance(self.matrix,np.ndarray):
                raise TypeError("Invalid tensor type or matrix type")

            if len(self.tensor.shape) != 3:
                raise ValueError("only accept 3-way tensor as input")
            if len(self.matrix.shape) != 2:
                raise ValueError("only accept 2-way matrix as input")
            

        except ValueError as ve:
            print("error:",ve)
        except TypeError as te:
            print("error:",te)
        
        