# -*- coding: utf-8 -*-
"""
@author: Abdul Hakmeh
@Email:ahak15@♣tu-claushtal.de
"""
import os
path= 'C:/anaconda3/envs/ahak15_projekt'
os.chdir(path)
import nilmtk
import matplotlib.pyplot as plt
import numpy as np
from nilmtk.utils import print_dict
from nilmtk.api import API
from IPython.display import display, HTML# display htmp display()
import warnings
warnings.filterwarnings("ignore")
from nilmtk.disaggregate import  Mean, CO 
from nilmtk.disaggregate.fhmm_exact import FHMMExact
from nilmtk.disaggregate.hart_85  import Hart85
import sys
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials,space_eval
import time
from nilmtk_contrib.disaggregate import Seq2Point, Seq2Seq
from hyperopt import hp
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
PYTHONHASHSEED=0 
from numba import cuda 
import csv
def call_NILMTK(params):    
    res={}
    algo= Seq2Seq if params['algorithm']== 's2s' else Seq2Point 
    experiment1 = {
      'power': {'mains': ['apparent'],'appliance': ['apparent']},
      'sample_rate':params['sample_rate'] ,
      'appliances':['microwave','fridge','cooker','washer dryer'],
      'methods': {      
        'Algorithm':algo({'n_epochs':params['n_epochs'],'batch_size':params['batch_size'],\
        'window':'none','history_aggregation_factor':params['history_aggregation_factor'], \
        'sequence_length':params['sequence_length'],\
        'time_spacing_function_value':params['time_spacing_function_value'],\
        'use_Hakmeh_windowing':params['use_Hakmeh_windowing'], 'spacing_function':params['spacing_function'],            
        'linear_function':params['linear_function'], 'sampling_function':params['sampling_function']  }),        
          },
      'train': {    
        'datasets': {
            'DRED': {
                'path': './dataset/DRED.h5',
                'buildings': {
                    1: {
                        'start_time': '2015-07-27',
                        'end_time': '2015-08-09'
                        }
                    }                
                }
            }
        },
      'test': {
        'datasets': {
            'DEED': {
                'path': './dataset/DRED.h5',
                'buildings': {
                    1: {
                        'start_time': '2015-08-10',
                        'end_time': '2015-08-16'
                        }
                    }
                }
            },
            'metrics':params['metrics']
        }         
          }              
    api_results_experiment_1 = API(experiment1)
    errors_keys = api_results_experiment_1.errors_keys
    errors_ = np.concatenate( api_results_experiment_1.errors , axis=0)
    res[(params['sequence_length'],params['history_aggregation_factor'],params['history_aggregation_factor'])]= errors_
    f1_avg= np.average(errors_[0:3])
    nde_avg=np.average(errors_[4:])
    errors_=np.concatenate(errors_[:])
    return   f1_avg,nde_avg,errors_
#########################################################################
#########################################################################

expriment_id= int(sys.argv[1])
algo= sys.argv[2]
time_spacing_function= str(sys.argv[3])
sampling_functions=sys.argv[4]
time_spacing_function_value=float(sys.argv[5])
factor=float(sys.argv[6])
linear_flag= True if sys.argv[7]=='TRUE' or sys.argv[7]=='True' else False
#print(linear_flag)
machine_id=sys.argv[8]
if sampling_functions == 'inverse':
    factor= 1
params= { 
 'ID':expriment_id,
 'algorithm':algo,
 'n_epochs' : 30 ,
 'sample_rate':  10,
 'batch_size' :512,
 'history_aggregation_factor':factor,
 'sequence_length' :561,
 'time_spacing_function_value':time_spacing_function_value ,
 'use_Hakmeh_windowing': True,
 'metrics': ['f1score','nde'],
 'spacing_function': time_spacing_function,
 'sampling_function':sampling_functions,
 'linear_function':linear_flag }
start_time = time.time()   
f1_avg,nde_avg,errors =call_NILMTK(params)
ex_time=(time.time() - start_time)
params['exec_Time(s)']=ex_time
#print('TIME: ',ex_time)
params['F1_AV']= f1_avg
params['NDE_AV']= nde_avg
device_metrics=['F1_MIC','F1_FRI','F1_COK','F1_WAM','NDE_MIC','NDE_FRI','NDE_COK','NDE_WAM']
for i in range(len(device_metrics)):              
       params[device_metrics[i]]= errors[i]          
################################## save result to csv file####################################
df = pd.DataFrame.from_dict(params,orient='index')
df = df.transpose()  
if sampling_functions=='inverse':
    res_path='./results/'+ algo +'_'+time_spacing_function +'_'+str(linear_flag)+'_'+ sampling_functions +'.csv' 
    if(os.path.isfile(res_path)):
        df.to_csv('./results/'+ algo+'_'+time_spacing_function +'_'+str(linear_flag)+'_'+ sampling_functions +'.csv',index=False,mode='a',header=False)
    else:
        df_ = pd.DataFrame.from_dict(params,orient='index')
        df_ = df_.transpose() 
        df_=df_.append(df)
        df_.to_csv('./results/'+ algo+'_'+time_spacing_function +'_'+str(linear_flag)+'_'+ sampling_functions +'.csv',index=False)
else: 
    res_path='./results/'+ algo +'_'+time_spacing_function +'_'+str(linear_flag)+'_'+ sampling_functions +'_'+str(time_spacing_function_value)+'.csv' 
    if(os.path.isfile(res_path)):
        df.to_csv('./results/'+ algo+'_'+time_spacing_function +'_'+str(linear_flag)+'_'+ sampling_functions +'_'+str(time_spacing_function_value)+'.csv',index=False,mode='a',header=False)
    else:
        df_ = pd.DataFrame.from_dict(params,orient='index')
        df_ = df_.transpose() 
        df_=df_.append(df)            
        df_.to_csv('./results/'+ algo+'_'+time_spacing_function +'_'+str(linear_flag)+'_'+ sampling_functions +'_'+str(time_spacing_function_value)+'.csv',index=False)
#################################update the status of the expriemnt######################
df = pd.read_csv('./Expriments_parameter/Machine_'+ machine_id + '_parameters.csv',header=None)
columns=np.linspace(1,len(df.columns),len(df.columns),endpoint=True,dtype=int)
df.columns=columns
df.loc[df[1] == expriment_id,8]='Done'
df.to_csv('./Expriments_parameter/Machine_'+ machine_id + '_parameters.csv',header=None,index=False)