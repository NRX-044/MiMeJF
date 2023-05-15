import numpy as np
import pandas as pd

## this class is mainly implemented for storing factor matrix

class factor_matrix:
    def __init__(self):
        self.tfm = None
        self.mfm = None
        self.weight = []
        self.variance = []
        self.metalist = {'microbe':[],
                        'metabolite':[],
                        'pathway':[],
                        'sample':[]}
    # naive implementation of khatri-rao product
    def kr4t_product(self,outer_idx):
        if self.tfm:
            oper_vector = self.tfm.copy()
            oper_vector.pop(outer_idx)
            prod = np.einsum("ir,jr -> ijr",
                             oper_vector[1],
                             oper_vector[0]).reshape(oper_vector[1].shape[0]*oper_vector[0].shape[0],-1)
            return prod
    def l2_norm(self):
    #normalize factor matrix loadings into unit length (remove the effect of different scales of variables)
    #past the norm to the weight
        for d in range(len(self.tfm)):
            scale = np.linalg.norm(self.tfm[d],
                                   2,
                                   axis=0)# l2 norm across each latent factor,length = number_of_latent_factors
            scale_nz = np.where(scale==0,
                                np.ones(self.tfm[d].shape[1],
                                        dtype=float),
                                scale)
        self.tfm[d]/=scale_nz
        mscale = np.linalg.norm(self.mfm,
                                2,
                                axis=0)
        self.mfm/=mscale

    def inf_norm(self):
    # perform factor matrix normalization in the order of inf norm 
    # input: a factor matrix object containing factor matrices of each mode

        for d in range(len(self.tfm)):

            # find the largest abosolute value in each column
            scale = np.linalg.norm(self.tfm[d],
                                   ord=np.inf,
                                   axis=0)
            self.tfm[d]/=scale
        mscale = np.linalg.norm(self.mfm,
                                ord=np.inf,
                                axis=0)
        self.mfm/=mscale
    
    # tagging the factor matrix with features 
    def get_metadata(self,
                     mi_list,
                     me_list,
                     pwy_list,
                     samp_list):

        self.metalist['microbe'] = mi_list
        self.metalist['metabolite'] = me_list
        self.metalist['pathway'] = pwy_list
        self.metalist['sample'] = samp_list
    
    # sample-phenotype mapper
    def type_map(self,
                tag_dict):
        try:
            if not self.metalist['sample']:
                raise ValueError("can not find list of samples, should attain metadata first\n")
            else:
                if set(tag_dict.keys()) != self.metalist['sample']:
                    raise ValueError("unmatched sample in tag dict\n")
        except ValueError as ve:
            print("error: ",ve)
        
        self.tag_dict = tag_dict

    
    #write the result to tab-separated file
    def to_tabfile(self,
                   outpath,
                   sep,
                   index = False):  
        try:
            if not self.tfm or not self.mfm:
                raise ValueError("can not find factor matrices, should perform the factorization first")
        except ValueError as ve:
            print("error: ",ve)
        
        if index:
            try:
                if not any(self.metalist.values()):
                    raise ValueError("metadata does not exist, should retrieve metadata first")
                for fm in self.tfm:
                    candidate_df = pd.DataFrame()
                    

            except ValueError as ve:
                print("error: ",ve)
        



        

