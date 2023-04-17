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


    
    