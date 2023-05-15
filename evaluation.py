import sys
import numpy as np
from scipy.stats import f
from utilities import refold
import copy
from optimization import *

## Implementation of quantification of the group separation of PCA-like results(only between two groups)
## input : feature distribution dataframe(sample x feature),sample labels (sample : label)
def separation_test(ftdist_df,final_sample_list,label_dict):
    all_label = list(set(label_dict.values()))
    sub_collection = []
    if len(all_label) > 2:
        print("type of labels cannot excess 2")
        sys.exit()
    
    for type_label in all_label:
        sub_sample = [x for x in final_sample_list if x in label_dict and label_dict[x] == type_label]
        sub_idx = [final_sample_list.index(x) for x in sub_sample]
        sub_ftdist = ftdist_df[sub_idx,:]
        sub_collection.append(sub_ftdist)

    diff_mat = np.mean(sub_collection[0],axis=0)-np.mean(sub_collection[1],axis=0)
    # it should be of shape (2,)

    
    spool_cov = 0
    for group in sub_collection:
        spool_cov+=np.cov(group.T)*(group.shape[0]-1)
    free_degree = np.sum([x.shape[0]-1 for x in sub_collection])
    spool_cov/=free_degree
    spool_cov_inv = np.linalg.inv(spool_cov)
    
    m_dist = np.matmul(np.matmul(diff_mat,spool_cov_inv.T),diff_mat.T)

    size1 = sub_collection[0].shape[0]
    size2 = sub_collection[1].shape[0]
    t_sqe = (size1*size2)/(size1+size2)*m_dist
    
    f_statistic = (size1+size2-sub_collection[0].shape[1]-1)/(sub_collection[0].shape[1]*(size1+size2-2))*t_sqe
    
    fdist = f(sub_collection[0].shape[1],size1+size2-sub_collection[0].shape[1]-1)
    
    p_val = 1-fdist.cdf(f_statistic)
    return p_val

def rmse_eval(fm,tensor,mat,mask=None):
    # use RMSE as loss function
    est_err = 0
    mode2_tenkai = np.matmul(fm.tfm[1],fm.kr4t_product(1).T)
    shape = list(tensor.shape)
    rc_tensor = refold(mode2_tenkai,1,shape)
    rc_matrix = np.matmul(fm.tfm[2],fm.mfm.T)
    if mask:
        idx_1 = np.nonzero(mask[0])
        idx_2 = np.nonzero(mask[1])
        est_err += np.linalg.norm(tensor[idx_1]-rc_tensor[idx_1])/np.sqrt(tensor[idx_1].size)
        est_err += np.linalg.norm(mat[idx_2]-rc_matrix[idx_2])/np.sqrt(mat[idx_2].size)
    else:
        est_err += np.linalg.norm(tensor-rc_tensor)/np.sqrt(tensor.size)
        est_err += np.linalg.norm(mat-rc_matrix)/np.sqrt(mat.size)

    return est_err

def r2_eval(fm,tensor,mat):
    # use R2 as loss function
    numerator = 0
    denominator = 0
    mode2_tenkai = np.matmul(fm.tfm[1],fm.kr4t_product(1).T) 
    shape = list(tensor.shape)

    rc_tensor = refold(mode2_tenkai,1,shape)

    numerator += np.linalg.norm(tensor-rc_tensor)
    
    rc_matrix = np.matmul(fm.tfm[2],fm.mfm.T)
    numerator += np.linalg.norm(mat-rc_matrix)

    total_norm = np.linalg.norm(tensor)+np.linalg.norm(mat)
    denominator+=total_norm
    r2 = 1 - numerator/denominator

    return r2

def sort_by_var(fac_mat,mm_tensor,mb_mat,final_var):
    # sort the latent component by explained variance
    ## method: In a for loop,rermove one component at a time, then calculate the ratio between total variance and component-reomoved
    ## model variance, bigger the ratio,the smaller variance that the removed-component can explain
    num_lf = fac_mat.tfm[0].shape[1]
    shape = [fac_mat.tfm[x].shape[0] for x in range(len(fac_mat.tfm))]
    rc_var = []
    
    mode2_tenkai = np.matmul(fac_mat.tfm[1],fac_mat.kr4t_product(1).T)

    rc_tensor = refold(mode2_tenkai,1,shape)
    rc_matrix = np.matmul(fac_mat.tfm[2],fac_mat.mfm.T)

    denominator = np.linalg.norm(rc_tensor) + np.linalg.norm(rc_matrix)

    ## the old method ###
    for d in range(num_lf):
        eval_fac_mat = copy.deepcopy(fac_mat)
        shape = [eval_fac_mat.tfm[x].shape[0] for x in range(len(eval_fac_mat.tfm))]

        eval_fac_mat.tfm = [np.delete(fm,d,axis=1) for fm in eval_fac_mat.tfm]
        eval_fac_mat.mfm = np.delete(fac_mat.mfm,d,axis=1)
        
        e_mode2_tenkai = np.matmul(eval_fac_mat.tfm[1],eval_fac_mat.kr4t_product(1).T)
        e_rc_tensor = refold(e_mode2_tenkai,1,shape)

        
        e_rc_matrix = np.matmul(eval_fac_mat.tfm[2],eval_fac_mat.mfm.T)
             
        numerator = np.linalg.norm(rc_tensor-e_rc_tensor) + np.linalg.norm(rc_matrix-e_rc_matrix)
        tmp_var = float(numerator)
        
        
        rc_var.append(tmp_var)
    
    fac_mat.variance = [1-float(tmp_var/denominator) for tmp_var in sorted(rc_var)]
    #fac_mat.variance = [float(tmp_var/np.sum(fac_mat.variance)) for tmp_var in fac_mat.variance]
    #fac_mat.variance = [float(final_var-tmp_var) for tmp_var in sorted(rc_var)]
    var_ord = np.argsort(rc_var)
    fac_mat.tfm = [fm[:,var_ord] for fm in fac_mat.tfm]
    fac_mat.mfm = fac_mat.mfm[:,var_ord]
    
    return fac_mat

# random cross validation for unsupervised method
def random_cv(sim_tensor,sim_matrix,tmask,mmask,lf,approach):
    
    inv_tmask = 1-tmask
    inv_mmask = 1-mmask
    tidx = np.nonzero(tmask)
    midx = np.nonzero(mmask)
    itidx = np.nonzero(inv_tmask)
    imidx = np.nonzero(inv_mmask)
    train_tensor = tmask * sim_tensor 
    train_matrix = mmask * sim_matrix



    shape = list(sim_tensor.shape)

    try:
        if approach != 'basic' or approach != 'aio':
            raise ValueError("factorization approach can only be chosen between basic and aio")
    except ValueError as ve:
        print("error: ",ve)
        sys.exit(1)
    #compute the RMSE of train set
    if approach == 'basic':
        decomp_fm = joint_mt(train_tensor,train_matrix,lf,tmask,mmask)
    elif approach == 'aio':
        #TODO 
        decomp_fm = all_in_one_mt(train_tensor,train_matrix,lf)
    
    re_trainmode2_tenkai = np.matmul(decomp_fm.tfm[1],decomp_fm.kr4t_product(1).T)
    re_train_tensor = refold(re_trainmode2_tenkai,1,shape)
    re_train_matrix = np.matmul(decomp_fm.tfm[2],decomp_fm.mfm.T)

    
    weight_tensor = sim_tensor.size/(sim_tensor.size+sim_matrix.size)
    weight_matrix = sim_matrix.size/(sim_tensor.size+sim_matrix.size)

    train_rce = weight_tensor*np.linalg.norm(sim_tensor[tidx]-re_train_tensor[tidx])/np.linalg.norm(sim_tensor[tidx])
    train_rce += weight_matrix*np.linalg.norm(sim_matrix[midx]-re_train_matrix[midx])/np.linalg.norm(sim_matrix[midx])

    
    #compute the RMSE of test set        
    test_rce = weight_tensor*np.linalg.norm(sim_tensor[itidx]-re_train_tensor[itidx])/np.linalg.norm(sim_tensor[itidx])
    test_rce += weight_matrix*np.linalg.norm(sim_matrix[imidx]-re_train_matrix[imidx])/np.linalg.norm(sim_matrix[imidx])

        
    return train_rce,test_rce