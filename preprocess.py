import os
import sys
import pandas as pd
import numpy as np
import glob

def tensor_process(file_list,
                   exclude_unclassified = False):
    
    # summarizing the total number of pathway and contributing microbes
    pwy_list = []
    mb_list = []
    sample_list = []
    for file in file_list:
        res = pd.read_csv(file,sep="\t",header = None)
        df = pd.DataFrame(res)

        
        for i in range(1,len(df)):
            tmp_feature = df.iloc[i,0].split("|")
            if len(tmp_feature) > 1:
                tmp_pwy = str(tmp_feature[0])
                if tmp_pwy != "UNMAPPED" and tmp_pwy != "UNINTEGRATED":
                    tmp_mb = str(tmp_feature[1])
                    if exclude_unclassified == False or tmp_mb != "unclassified":
                        if tmp_pwy not in pwy_list:
                            pwy_list.append(tmp_pwy)
                        if tmp_mb not in mb_list:
                            mb_list.append(tmp_mb)
            
    pwy_set = dict(zip(pwy_list,[x for x in range(len(pwy_list))]))
    mb_set = dict(zip(mb_list,[x for x in range(len(mb_list))]))

    tensor = np.zeros((len(pwy_list),len(mb_list),len(file_list)),dtype=float)
    # generating the pathway-microbe matrix for each sample 
    file_idx = 0
    for file in file_list:
        res=pd.read_csv(file,sep="\t",header=None)
        df = pd.DataFrame(res)
        filename = file.split("\\")[-1]
        samplename = filename.replace("_pathabundance.tsv","")
        sample_list.append(samplename) 
      
        for i in range(1,len(df)):
            tmp_feature = df.iloc[i,0].split("|")
            if len(tmp_feature) > 1:
                tmp_pwy = str(tmp_feature[0])
                if tmp_pwy != "UNMAPPED" and tmp_pwy != "UNINTEGRATED":
                    tmp_mb = str(tmp_feature[1])
                    if exclude_unclassified == False or tmp_mb != "unclassified":
                        tmp_relab = float(df.iloc[i,1])
                        tensor[pwy_set[tmp_pwy],mb_set[tmp_mb],file_idx] = tmp_relab
        file_idx+=1
    
    return tensor,pwy_list,mb_list,sample_list

def metabol_extract(mbpath,
                    mapperpath,
                    sample_list,
                    output_label,
                    metrans):
    mb_file = pd.read_csv(mbpath,sep = "\t",header = 0,index_col=0)
    mb_df = pd.DataFrame(mb_file)
    metabolite_list = list(mb_df.index)
    
    map_file = pd.read_csv(mapperpath,sep = "\t",header = 0)
    map_df = pd.DataFrame(map_file)
    
    strg_miid = [str(x) for x in list(map_df['misamp_id'])]
    strg_meid = [str(x) for x in list(map_df['mesamp_id'])]
    samp_label = [str(x) for x in list(map_df['type'])]
    

    ss_dict = dict(zip(strg_miid,strg_meid))
    new_sample_list = [ss_dict[x] for x in sample_list]

    # reorder the sample in the order of microbiome functional profile
    new_mb_df = reorder(mb_df,new_sample_list,metrans)
    try:
        if output_label == 'metabolite':
            final_sample_list = new_sample_list
            label_dict = dict(zip(strg_meid,samp_label))
        elif output_label == 'microbe':
            final_sample_list = sample_list
            label_dict = dict(zip(strg_miid,samp_label))            
        else:
            raise ValueError("invalid data type selected when determining output labels\n")
    except ValueError as ve:
        print("error:",ve)
    return new_mb_df,final_sample_list,label_dict,metabolite_list

def reorder(metabol_df,
            sample_list,
            metrans):
    try:
        ori_sample_list = list(metabol_df.columns)        
        if set(ori_sample_list) == set(sample_list):
            metabol_df = metabol_df[sample_list]
            candidate_mb = metabol_df.values
            if metrans == True:
                zto_candidate_mb = np.where(candidate_mb==0,
                                            1,
                                            candidate_mb)
                candidate_df = mb_preprocess(zto_candidate_mb.T)
            else:
                candidate_df = candidate_mb.T
            return candidate_df       
        else:
            raise ValueError("elements in metabolomics file must equal to microbiome functional file\n")
    except ValueError as ve:
        print("Critical error:",ve)

## clr transformation for metabolome data
def mb_preprocess(mb_df):
    def clr_trans(x,f_list):
        new_x = 0
        for e in f_list:
            if e != x:
                new_x+=np.log(x/e)
        new_x = new_x/len(f_list)
        return new_x
   
    trans_mb_list = []
    feature_size = mb_df.shape[1]
    for i in range(mb_df.shape[0]):
        new_vector = [clr_trans(x,mb_df[i]) for x in mb_df[i]]
        trans_mb_list.append(new_vector)
    trans_mb_df = np.array(trans_mb_list)
    return trans_mb_df

#log scaling of total functional pathway abundance(followed by linear scaling of stratified species)
def tensor_logscale(mm_tensor):
    for i in range(mm_tensor.shape[0]):
        for j in range(mm_tensor.shape[2]):
            total_ab = sum(mm_tensor[i,:,j])
            if total_ab == 0:
                continue
            scale_fac = np.log(total_ab)
            tmp_contrib = np.array([x/total_ab*scale_fac for x in mm_tensor[i,:,j]])
            mm_tensor[i,:,j] = tmp_contrib
    return mm_tensor


# main pipeline for data construction
def data_construct(fpdir,
                   mtfile,
                   mapperfile,
                   exclude_unclassified=False,
                   output_label='metabolite',
                   mitrans=True,
                   metrans=True):
    
    #file format & existence checking
    try:
        map_file = pd.read_csv(mapperfile,sep='\t',header=0)
        map_df = pd.DataFrame(map_file)
        map_col = map_df.columns.tolist()
        if set(map_col) != set(['misamp_id','mesamp_id','type']):
            raise ValueError("invalid term in header of metadata file\n")
        
        fpfile_list = list(glob.glob(os.path.join(fpdir,'*.tsv')))
        if not fpfile_list:
            raise ValueError('empty directory detected')
    except ValueError as ve:
        print("error:",ve)
        sys.exit(1)

    fc_tensor,pwy_list,mb_list,sample_list = tensor_process(fpfile_list,
                                                            exclude_unclassified)
    if mitrans:
        fc_tensor = tensor_logscale(fc_tensor)

    fc_matrix,final_sample_list,label_dict,metabolite_list = metabol_extract(mtfile,mapperfile,
                                                                             sample_list,
                                                                             output_label = output_label,
                                                                             metrans=metrans)
    
    data_dict = {'tensor':fc_tensor,
                 'matrix':fc_matrix,
                 'pathway':pwy_list,
                 'microbe':mb_list,
                 'metabolite':metabolite_list,
                 'sample':final_sample_list,
                 'sample-type mapper':label_dict}
    return data_dict
