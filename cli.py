import os
import sys
import re
from utilities import *
import pandas as pd
import glob
import argparse
from mmjf import BasicFac, StructRevealFac
from preprocess import *




def main():
    # can be used as command line tools
    parser = argparse.ArgumentParser('mfpm is a tool capable of joint analysis of microbiome and metabolomic data')
    parser.add_argument('-i1','--input1',
                        required=True,
                        help='the folder containing microbiome functional profiling files')
    parser.add_argument('-i2',
                        '--input2',
                        required=True,
                        help='the metabolome table,where the columns are samples and the rows are metabolites')
    parser.add_argument('-m',
                        '--meta',
                        required=True,
                        help='the metadata table file, containing sample ID and their attributes respectively')
    parser.add_argument('-e',
                        '--exclude',
                        default=False,
                        help='whether exclude unclassified microbes in microbiome functional profiling data,default is True')
    parser.add_argument('-f',
                        '--factor',
                        default = 3,
                        help = 'the number of latent factors,default is 3')
    parser.add_argument('-l',
                        '--label',
                        default='metabolite',
                        help='choose sample id from either metabolome data or microbiome data,default is metabolome data')
    parser.add_argument('-mdl',
                        '--model',
                        default = 'basic',
                        help = 'choose the CMTF model between basic and structure revealing one, default is basic')
    parser.add_argument('-mitr',
                        '--mitrans',
                        default=True,
                        help='whether perform log transformation for microbiome functional profile,default is True')
    parser.add_argument('-metr',
                        '--metrans',
                        default=True,
                        help='whether perform centered log ratio transformation for metabolome data,default is True')
    parser.add_argument('-o',
                        '--output',
                        default=os.getcwd(),
                        help='output directory,default is current directory')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        try:
            os.makedirs(args.output)
        except:
            print('could not create a directory\n')
            print('check you input path again\n')
            print('process halted\n')
            sys.exit(1)

    match_slash = re.compile('/')
    match_antislash = re.compile('\\\\')

    path = re.sub('\\$','',args.input1)
    path = re.sub('/$','',path)
    match1 = re.search(match_slash,path)
    match2 = re.search(match_antislash,path)
    if match1:
        fpdir = path + '/'
    elif match2:
        path = path + '\\'
        fpdir = path.replace('\\\\','\\')
    mbpath = args.input2
    mapperpath = args.meta

    try:
        if not os.path.exists(fpdir):
            raise FileNotFoundError(f'can not find such directory {fpdir}')
        if not os.path.exists(mbpath):
            raise FileNotFoundError(f'can not find metabolites file {mbpath}')
        if not os.path.exists(mapperpath):
            raise FileNotFoundError(f'can not find metadata file {mapperpath}')
    except FileNotFoundError:
        print('Error:',FileNotFoundError)
        sys.exit(1)


    exclude_unclassified = args.exclude
    num_lf = args.factor
    output_label = args.label
    mitrans = args.mitrans
    metrans = args.metrans
    input_data = data_construct(fpdir,
                                mbpath,
                                mapperpath,
                                exclude_unclassified,
                                output_label,
                                mitrans,metrans)
    c_tensor = input_data['tensor']
    c_matrix = input_data['matrix']
    mi_list = input_data['microbe']
    pwy_list = input_data['pathway']
    me_list = input_data['metabolite']
    samp_list = input_data['sample']
    tag_dict = input_data['sample-type mapper']


    mdl = args.model
    try:
        if mdl not in set(['basic','structure revealing']):
            raise ValueError('unknown model,can only be chosen between basic and structure revealing\n')
        elif mdl == 'basic':
            fac_process = BasicFac(c_tensor,c_matrix,num_lf)
            fac_process.factorization()
        elif mdl == 'structure revealing':
            fac_process = StructRevealFac(c_tensor,c_matrix,num_lf)
        fac_process.get_metadata(mi_list,me_list,pwy_list,samp_list)
        fac_process.type_map(tag_dict)
        

            
    except ValueError as ve:
        print("error: ",ve)
    

    #TODO
if __name__=='__main__':
    main()





