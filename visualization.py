import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# quick plotting function of feature loadings
def feature_plot(decomp_fm,
                 fttag,
                 feature_list,
                 important_ft,
                 outpath):
    plt.style.use('fivethirtyeight')
    
    for lf in range(decomp_fm.lf):
        if decomp_fm.variance:
            var_exp = decomp_fm.variance[lf]*100
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
        
        if fttag < 2:
            total_ftdist = list(decomp_fm.tfm[fttag][:,lf])
        else:
            total_ftdist = list(decomp_fm.mfm[:,lf])
        try:
            total_idx = [x for x in range(len(feature_list))]
            if len(total_ftdist) != len(total_idx):
                raise ValueError("unmatched length between feature list and factor matrix\n")
            cosort_arr = zip(total_ftdist,total_idx)
            sorted_ftdist = [x for x,i in sorted(cosort_arr,key=lambda p:abs(p[0]),reverse=True)][:30]
            cosort_arr = zip(total_ftdist,total_idx)
            sorted_idx = [i for x,i in sorted(cosort_arr,key=lambda p:abs(p[0]),reverse=True)][:30]
            if fttag == 0:
                sub_ft_list = [feature_list[x].split(":")[-1] for x in sorted_idx]
            elif fttag == 1:
                sub_ft_list = [feature_list[x].split(".s__")[-1] for x in sorted_idx]
            elif fttag == 2:
                sub_ft_list = [feature_list[x] for x in sorted_idx]

        except ValueError as ve:
            print("error:",ve)
        min_bound = min(total_ftdist)-0.2    
        max_bound = max(total_ftdist)+0.2
        ax.set_ylim(min_bound,max_bound)
        
           
        ax.set_xticks([x for x in range(len(sub_ft_list))])
        ax.set_xticklabels(sub_ft_list,rotation = 45,fontsize = 10)
        color_idx = ['black' for x in range(len(sub_ft_list))]
        
        for idx,pre_ft in enumerate(sub_ft_list):
            if pre_ft in important_ft:
                color_idx[idx] = 'red'

            

                    
        
        for ticklabel,tickcolor in zip(ax.get_xticklabels(),color_idx):
            ticklabel.set_color(tickcolor)
        
        x_coor = [x for x in range(len(sub_ft_list))]
        
        ax.bar(x_coor,sorted_ftdist,color = 'blue')
        
        
        plt.setp(ax.get_xticklabels(),
                 rotation=45,
                 ha="right",
                 rotation_mode="anchor",
                 fontsize=10)
        if decomp_fm.variance:
            ax.set_title(f'variance explained: {var_exp}%')
        elif decomp_fm.weight:
            ax.set_title(f'tensor weight: {decomp_fm.weight[0][lf]} matrix weight: {decomp_fm.weight[1][lf]}',fontsize=10)
        plt.tight_layout()
        plt.show()
        if fttag == 0:
            keyword = 'pathway'
        elif fttag == 1:
            keyword = 'microbe'
        elif fttag == 2:
            keyword = 'metabolite'
        
        outpath = re.sub('/$','',outpath)
        outpath = outpath + '/'
        savepath = outpath+keyword+'_top30inlf'+str(lf+1)+".pdf"
        fig.savefig(savepath,dpi=800)


def sample_plot(decomp_fm,
                samp_list,
                label_dict,
                outpath):
    plt.style.use("fivethirtyeight")
    fig = plt.figure(figsize=(10,10))
    color_box = ['#37AB65', '#3DF735', '#AD6D70', '#EC2504', '#8C0B90', '#C0E4FF', '#27B502', '#7C60A8', '#CF95D7', '#145JKH']
    group_type = list(set(label_dict.values()))   
    for lf in range(decomp_fm.lf):
        if decomp_fm.variance:
            var_exp = decomp_fm.variance[lf]*100
        ax = fig.add_subplot(3,1,lf+1)
        ax.set_xticks([x for x in range(len(samp_list))])

        ax.set_xticklabels(samp_list,rotation=45,fontsize =  10)
        print(len(samp_list))
        if decomp_fm.weight:
            print(f"present the weight of lf {lf} in microbiome functional profiling tensor: ")
            print(decomp_fm.weight[0][lf])
            print(f"present the weight of lf {lf} in metabolites profiling matrix: ")
            print(decomp_fm.weight[1][lf])
        
        candidate_ftdist = decomp_fm.tfm[2][:,lf]
  
        min_bound = min(candidate_ftdist)-0.2    
        max_bound = max(candidate_ftdist)+0.2
        
        ax.set_ylim(min_bound,max_bound)
        
        for idx in range(len(group_type)):
            sub_sample = [x for x in samp_list if label_dict[x] == group_type[idx]]
            x_coor = [samp_list.index(x) for x in sub_sample]
            sub_ftdist = candidate_ftdist[x_coor]
            ax.scatter(x_coor,sub_ftdist,color=color_box[idx],alpha=0.5,label = group_type[idx])
        
        if decomp_fm.variance:
            ax.set_title(f'variance explained: {var_exp}%',fontsize=10)
        elif decomp_fm.weight:
            ax.set_title(f'tensor weight: {decomp_fm.weight[0][lf]} matrix weight: {decomp_fm.weight[1][lf]}',fontsize=10)
        ax.legend(loc = 'upper right',fontsize=10)
    plt.tight_layout()
    plt.show()

    outpath = re.sub('/$','',outpath)
    outpath = outpath + '/'
    savepath = outpath + '/' + 'samp_dist.pdf'
    fig.savefig(savepath,dpi=800)

    
    