import sys
import numpy as np
import scipy.optimize  as sopt
from utilities import tenvec,raw_krp,refold
from FactorMatrix import factor_matrix
from evaluation import rmse_eval,sort_by_var

# alternating least squares for optimizing coupled matrix and tensor factorization model
def als_optimize(fm,
                 tensor_unfold,
                 mb_df=None,
                 idx=None):
    V = np.array([])
    if idx != None:
        for d in range(len(fm.tfm)):
            if d != idx:
                if V.size != 0:
                    V = np.matmul(fm.tfm[d].T,fm.tfm[d])*V
                else:
                    V = np.matmul(fm.tfm[d].T,fm.tfm[d])

        comp_unfold = tensor_unfold[idx]
        kr_product = fm.kr4t_product(idx).T
        tensorshape = [lm.shape[0] for lm in fm.tfm]
        #M1 = big_mat_product_compute(fm,tensorshape,idx,tensor_unfold)
        if idx == 2:
            comp_unfold = np.hstack((tensor_unfold[idx],mb_df))
            kr_product = np.hstack((kr_product,fm.mfm.T))
            #M2 = np.matmul(mb_df,fm.mfm)
            #M1 = np.hstack(M1,M2)
            V = V + np.matmul(fm.mfm.T,fm.mfm)
        pinverseV = np.matmul(np.linalg.pinv(np.matmul(V.T,V)),V.T)
        F = np.matmul(np.matmul(comp_unfold,kr_product.T),pinverseV)
        return F
    else:
        M = np.matmul(mb_df.T,np.linalg.pinv(fm.tfm[2]).T)
        return M

# generate objective function and gradients for nonlinear conjugate gradient optimization
def funcgrad_gen(mm_tensor,
                 mb_mat,
                 fm):
    #alpha param for feature weights,PROBABLY 1 is a decent choice
    # beta: sparsity parameter for weight of components
    # eps: a small constant which helps computing differentiation 
    
    #--------------------------------------
    # microbiome function profiling tensor of shape (pathway,microbe,sample)
    # metabolome matrix of shape (sample,metabolite)

    
    def obj_f(x,*args):
        mm_tensor,mb_mat,alpha,beta_ten,beta_mat,eps,mask,lf = args
        shape = list(mm_tensor.shape)
        mshape = list(mb_mat.shape)
        
        tfm = [np.zeros((shape[x],lf),dtype=float) for x in range(len(shape))]
        #rebuild all factor matrices
        cnt_idx = 0
        for d in range(len(shape)):
            if d < 1:
                tfm[d] = x[0:(shape[d]*lf)].reshape(shape[d],lf)
            else:
                cnt_idx+=shape[d-1]*lf
                tfm[d] = x[cnt_idx:(cnt_idx+shape[d]*lf)].reshape(shape[d],lf)
        cnt_idx+=shape[-1]*lf
        mfm = x[cnt_idx:(cnt_idx+mshape[-1]*lf)].reshape(mshape[-1],lf)
        cnt_idx+=mshape[-1]*lf
        tweight = x[cnt_idx:cnt_idx+lf].reshape(1,lf)
        cnt_idx+=lf
        mweight = x[cnt_idx:cnt_idx+lf].reshape(1,lf)        
        
        #reconstruct tensor by factor matrices and weights of components
        mode2_tenkai = np.matmul(tfm[1]*tweight,raw_krp(tfm[2],tfm[0]).T)
        rc_tensor =refold(mode2_tenkai,1,shape)
        #reconstruct matrix by factor matrices and weights of components
        rc_matrix = np.matmul(tfm[2]*mweight,mfm.T)
        if mask:
            rc_tensor = mask[0]*rc_tensor
            rc_matrix = mask[1]*rc_matrix
            
        #add 0.5 multiplication for derivative computation efficiency
        obj_func = 0.5*np.square(np.linalg.norm(mm_tensor))
        -np.sum(np.multiply(mm_tensor,rc_tensor)) 
        + 0.5*np.square(np.linalg.norm(rc_tensor))
        obj_func+= 0.5*np.square(np.linalg.norm(mb_mat))
        -np.sum(np.multiply(mb_mat,rc_matrix)) 
        + 0.5*np.square(np.linalg.norm(rc_matrix))

        #adding constraints of weights in objective function
        for r in range(tfm[0].shape[1]):
            obj_func+=0.5*beta_ten*np.sqrt(np.square(tweight[0,r])+eps)
            +0.5*beta_mat*np.sqrt(np.square(mweight[0,r])+eps)
        #adding norm constraints of each feature in objective function
        all_fm = [tfm[0],tfm[1],tfm[2],mfm]
        for ft in range(len(all_fm)):
            for r in range(tfm[0].shape[1]):
                obj_func+=0.5*alpha*np.square(np.linalg.norm(all_fm[ft][:,r])-1)
        return obj_func
    #gradient expression
    #each element except last two in grad is a numpy array of shape(feature_length,r),the last two elements are  arrays of shape(,r)
       
    def total_grad(x,*args):
        mm_tensor,mb_mat,alpha,beta_ten,beta_mat,eps,mask,lf = args
        shape = list(mm_tensor.shape)
        mshape = list(mb_mat.shape)
        
        tfm = [np.zeros((shape[x],lf),dtype=float) for x in range(len(shape))]
        #rebuild all factor matrices
        cnt_idx = 0
        for d in range(len(shape)):
            if d < 1:
                tfm[d] = x[0:(shape[d]*lf)].reshape(shape[d],lf)
            else:
                cnt_idx+=shape[d-1]*lf
                tfm[d] = x[cnt_idx:(cnt_idx+shape[d]*lf)].reshape(shape[d],lf)
        cnt_idx+=shape[-1]*lf
        mfm = x[cnt_idx:(cnt_idx+mshape[-1]*lf)].reshape(mshape[-1],lf)
        cnt_idx+=mshape[-1]*lf
        tweight = x[cnt_idx:cnt_idx+lf].reshape(1,lf)
        cnt_idx+=lf
        mweight = x[cnt_idx:cnt_idx+lf].reshape(1,lf)
        
        
        # feature facor matrices gradient
        ft_grad = [np.zeros((shape[x],tfm[0].shape[1]),dtype=float) for x in range(len(shape))]
        
        mode2_tenkai = np.matmul(tfm[1]*tweight,raw_krp(tfm[2],tfm[0]).T)
        rc_tensor =refold(mode2_tenkai,1,shape)
        rc_matrix = np.matmul(tfm[2]*mweight,mfm.T)
        for ft in range(len(shape)):
            diff_unfold = np.reshape(np.moveaxis(mm_tensor-rc_tensor,ft,0),((mm_tensor-rc_tensor).shape[ft],-1),order='F')
            tmp_idx = [0,1,2]
            tmp_idx.pop(ft)
            kr_prod = raw_krp(tfm[tmp_idx[1]],tfm[tmp_idx[0]])
            kr_prod = raw_krp(tweight,kr_prod)
            ft_grad[ft] = -np.matmul(diff_unfold,kr_prod)
            if ft == 2:
                diff_matrix = mb_mat-rc_matrix
                ft_grad[ft] -= np.matmul(diff_matrix,np.matmul(mfm,np.diag(np.squeeze(mweight))))
            ft_grad[ft]+=alpha*(tfm[ft]-(tfm[ft]/np.linalg.norm(tfm[ft],axis=0)))
        # compute gradient of another feature in mb matrix 
        diff_matrix = (mb_mat-rc_matrix).T
        ft_grad.append(-np.matmul(diff_matrix,np.matmul(tfm[2],np.diag(np.squeeze(mweight)))))
        ft_grad[-1]+=alpha*(mfm-(mfm/np.linalg.norm(mfm,axis=0)))

        #compute the gradient of weights     
        wgt_grad = []
        wgt_lambda = np.zeros((1,tfm[0].shape[1]),dtype=float)
        for r in range(tfm[0].shape[1]):
            diff_tensor = mm_tensor-rc_tensor
            vector_list = [tfm[x][:,r] for x in range(len(shape))]
            mode = [0,1,2]
            wgt_lambda[0,r]-=tenvec(diff_tensor,vector_list,mode)
            wgt_lambda[0,r]+= 0.5*beta_ten*tweight[0,r]*np.sqrt(np.square(tweight[0,r])+eps)
        wgt_grad.append(wgt_lambda)# a vector of shape(r,1)
    
        wgt_sigma = np.zeros((1,tfm[0].shape[1]),dtype=float)
        for r in range(tfm[0].shape[1]):
            diff_matrix = rc_matrix-mb_mat
            v4mat_list = [tfm[2][:,r],mfm[:,r]]
            mode2 = [0,1]
            wgt_sigma[0,r]+=tenvec(diff_matrix,v4mat_list,mode2)
            wgt_sigma[0,r]+=0.5*beta_mat*mweight[0,r]*np.sqrt(np.square(mweight[0,r])+eps)
        wgt_grad.append(wgt_sigma) # a vector of shape(r,1)

    
        #concatenate all gradients into a numpy array
        all_grad  = np.array([])
        for d in range(len(ft_grad)):
            if not all_grad.size:
                all_grad = ft_grad[d].ravel()
            else:
                all_grad = np.append(all_grad,ft_grad[d].ravel())
        for d in range(len(wgt_grad)):
            all_grad = np.append(all_grad,wgt_grad[d].ravel())
        
        return all_grad
           
    return obj_f,total_grad

#prototype function for coupling microbes-predicted functional pathways 3-order tensor and metabolites matrix
#return normalized unit length factor matrices of each dimensions and corresponding weight.
def joint_mt(mm_tensor,
             mb_mat,
             lf,
             tmask=np.array([]),
             mmask=np.array([])):
    #first initialize factorized rank-1 tensors and matrices using SVD

    init_fm = factor_matrix()
    init_fm.tfm = []

    if tmask.size and mmask.size:
        mask = [tmask,mmask]
    else:
        mask = []

    for d in range(len(mm_tensor.shape)):
        m_unfold = np.reshape(np.moveaxis(mm_tensor,d,0),(mm_tensor.shape[d],-1),order='F')
        
        if d == 2:        
            m_unfold = np.hstack((m_unfold,mb_mat))

        # initialization of factor matrices based on SVD

        sub_eima = np.linalg.svd(m_unfold,full_matrices=False)[0]
        if lf <= sub_eima.shape[1]:
            init_fm.tfm.append(sub_eima[:,:lf])
        else:
            rnd_fm = np.random.rand(sub_eima.shape[0],lf-sub_eima.shape[1])
            init_fm.tfm.append(np.hstack((sub_eima,rnd_fm)))
    # pre unfold tensor for downstream computation
    ori_unfold = []
    for d in range(len(mm_tensor.shape)):
        tmp_unfold = np.reshape(np.moveaxis(mm_tensor,d,0),(mm_tensor.shape[d],-1),order='F')
        ori_unfold.append(tmp_unfold)
    print("pre-unfold completed")
    #initialize the factor matrix of original matrix
    init_fm.mfm = als_optimize(init_fm,ori_unfold,mb_df=mb_mat,idx=None)


    var_est = []
    if mask:
        init_var = rmse_eval(init_fm,mm_tensor,mb_mat,mask)
    else:
        init_var = rmse_eval(init_fm,mm_tensor,mb_mat)
    var_est.append(init_var)
    #iterating the optimizing procedure until the loss function reaches converge
    for i in range(1,2000+1):    
        # minimizing each mode of tensor(including sample mode)
        for d in range(len(mm_tensor.shape)):
            init_fm.tfm[d] = als_optimize(init_fm,ori_unfold,mb_df=mb_mat,idx=d)
        # minimizing matrix's extra info factor matrix
        init_fm.mfm = als_optimize(init_fm,ori_unfold,mb_df=mb_mat,idx=None)
        
        #normalize factor matrices and get weights after a full iteration on all dimensions
        #fac_norm(init_fm)
        if mask:
            var = rmse_eval(init_fm,mm_tensor,mb_mat,mask)
        else:
            var = rmse_eval(init_fm,mm_tensor,mb_mat)
        var_est.append(var)
        print(f"round {i} completed")
        if abs((var_est[i]-var_est[i-1])/var_est[i-1]) <= 1e-6 or i==2000:
            print(f"round {i} reached convergence or reached max iteration")
            final_var = var_est[i]
            break
    

    #init_fm = sort_by_weight(init_fm)
    init_fm = sort_by_var(init_fm,mm_tensor,mb_mat,final_var)
    #init_fm = fac_norm(init_fm)
    init_fm.inf_norm()
    
    return init_fm

## another method for coupling microbes-predicted functional pathways 3-order tensor and metabolites matrix
## Reference:Acar, E., Papalexakis, E.E., Gürdeniz, G. et al. Structure-revealing data fusion. 
## BMC Bioinformatics 15, 239 (2014). https://doi.org/10.1186/1471-2105-15-239
def all_in_one_mt(mm_tensor,
                  mb_mat,
                  lf,
                  cparam,
                  tmask=np.array([]),
                  mmask=np.array([])):
    #get the initial values first
    try: 
        if len(mm_tensor.shape) != 3:
            raise ValueError("only accept 3-way tensor as input")
    except ValueError as ve:
        print("error:",ve)

    if tmask.size and mmask.size:
        mask = [tmask,mmask]
    else:
        mask = []

    init_fm = factor_matrix()
    init_fm.tfm = []
    #SVD-based factor matrices initialization
    for d in range(len(mm_tensor.shape)):
        tmp_unfold = np.reshape(np.moveaxis(mm_tensor,d,0),(mm_tensor.shape[d],-1),order='F')
        if d == 2:        
            tmp_unfold = np.hstack((tmp_unfold,mb_mat))
        sub_eima = np.linalg.svd(tmp_unfold,full_matrices=False)[0]
        if lf <= sub_eima.shape[1]:
            init_fm.tfm.append(sub_eima[:,:lf])
        else:
            rnd_fm = np.random.rand(sub_eima.shape[0],lf-sub_eima.shape[1])
            init_fm.tfm.append(np.hstack((sub_eima,rnd_fm)))
    uniq_mat4fac = np.linalg.svd(mb_mat.T,full_matrices=False)[0]
    if lf <= uniq_mat4fac.shape[1]:
        init_fm.mfm = uniq_mat4fac[:,:lf]
    else:
        rnd_fm = np.random.rand(uniq_mat4fac.shape[0],lf-uniq_mat4fac.shape[1])
        init_fm.mfm = np.hstack((uniq_mat4fac,rnd_fm))
        
    # weight initialization
    for i in range(2):
        init_fm.weight.append(np.ones((1,lf),dtype=float))
    
    x0 = np.array([])
    for d in range(len(mm_tensor.shape)):
        if not x0.size:
            x0 = init_fm.tfm[d].ravel()
        else:
            x0 = np.append(x0,init_fm.tfm[d].ravel())
    x0 = np.append(x0,init_fm.mfm.ravel())

    for i in range(2):
        x0 = np.append(x0,init_fm.weight[i].ravel())

    
    
    # generate objective function and corresponding gradients 
    func,grad = funcgrad_gen(mm_tensor,mb_mat,init_fm,mask)
    print("obj function and gradient generation completed\n")
    
    
    
    
    # the constant argument in model
    ALPHA = cparam['alpha']
    BETA_TEN = cparam['beta_ten']
    BETA_MAT = cparam['beta_mat']
    EPS = cparam['eps']
    ARGS = (mm_tensor,mb_mat,ALPHA,BETA_TEN,BETA_MAT,EPS,mask,lf)

    OPTS = {'maxiter' : None,
            'disp' : True,
            'gtol' : 1e-8,
            'norm' : np.inf,
            }
    #performing nonlinear conjugate gradient optimization to compute factor matrices and their weights
    print("started BFGS optimization...\n")
    all_res = sopt.minimize(func,x0,jac=grad,args=ARGS,method="BFGS",options=OPTS)
    print("Optimization Completed\n")
    return all_res

