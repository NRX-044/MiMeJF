import numpy as np


def refold(unfold_t,
           mode,
           shape):
    ## unfold_t:a mode-n unfolded tensor ,in the form of matrix
    ## mode:an integer that represents the mode of unfolding
    ## shape:a list that represents the original shape of 3-way tensor
        
    if mode == 0:
        recontruct_t = np.reshape(unfold_t,tuple(shape),order='F')
        return reconstruct_t
    elif mode == 1:
        shape[0],shape[1] = shape[1],shape[0]
        tmp_reconstruct_t = np.reshape(unfold_t,tuple(shape),order='F')
        re_list = []
        for d in range(shape[2]):
            re_list.append(tmp_reconstruct_t[:,:,d].T)
        reconstruct_t = np.dstack(tuple(re_list))
        return reconstruct_t
    elif mode == 2:
        tmp_reconstruct_t = unfold_t.T
        t_dim = shape[2]
        new_shape = shape.copy()
        new_shape.pop()
        re_list = []
        for d in range(t_dim):
            re_list.append(np.reshape(tmp_reconstruct_t[:,d],tuple(new_shape),order='F'))
        reconstruct_t = np.dstack(tuple(re_list))
        return reconstruct_t
    else:
        print("mode excesses the number of dimensions of tensor")

def raw_krp(mat1,
            mat2):
    # can compute khatri-rao product of arbitary pair of matrices 
    prod = np.einsum('ir,jr->ijr',mat1,mat2).reshape(mat1.shape[0]*mat2.shape[0],-1)
    return prod

#python implementation for computing product of tensor and vector(only for 3-ways tensor)
# Reference: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.3, www.tensortoolbox.org, August 16, 2022
def tenvec(tensor,
           vector_list,
           mode):
    ## argument tensor: 3-way tensor with the shape of (m,n,k)
    ## argument vector_list: list of vector that is used to multiplied
    ## argument mode:the mode involved in multiplication 
    
    #transform all vectors shape into (x,)
    op_vector_list = []
    try:
        for vec in vector_list:
            if isinstance(vec,np.ndarray):
                tmp_vec = np.reshape(vec,(vec.shape[0],),order='F')
                op_vector_list.append(tmp_vec)
            elif isinstance(vec,list):
                tmp_vec = np.reshape(vec,(len(vec),),order='F')
                op_vector_list.append(tmp_vec)
            else:
                raise TypeError("unsupported data type of vector")
    except TypeError as te:
        print("error",te)
    #check argument consistency
    try:
        for m in mode:
            if m not in np.arange(len(tensor.shape)):
                raise ValueError("number of dimensions in tensor does not match the modes involved")
                break
        if len(mode) != len(op_vector_list):
            raise ValueError("number of modes involved does not match the number of vectors)")
        for i,m in enumerate(mode):
            if op_vector_list[i].shape[0] != tensor.shape[m]:
                raise ValueError("vector length does not match the corresponding tensor dimensions")
    except ValueError as ve:
        print("error:",ve)

    
    # sort mode and vector list from highest idx to lowest
    combine_list = zip(op_vector_list,mode)
    sorted_mode = [i for x,i in sorted(combine_list,key = lambda x:x[1],reverse=True)]
    combine_list = zip(op_vector_list,mode)
    op_vector_list = [x for x,i in sorted(combine_list,key = lambda x:x[1],reverse=True)]
     
    # 3-ways tensors only 
    tenvec_xp = np.array([])
    cnt = 1
    for i,d in enumerate(sorted_mode):
        if cnt == 1:
            tmp_tenkai = np.reshape(np.moveaxis(tensor,d,0),
                                    (tensor.shape[d],-1),
                                    order='F')
            tenvec_xp = np.matmul(op_vector_list[i].T,tmp_tenkai)
            cnt+=1
        else:
            tenvec_xp = np.reshape(tenvec_xp,
                                   (int(np.prod(tensor.shape[0:d])),
                                    int(tensor.shape[d])),
                                    order='F')
            tenvec_xp = np.squeeze(tenvec_xp)
            tenvec_xp = np.matmul(tenvec_xp,op_vector_list[i].T)
            cnt+=1
    return tenvec_xp


