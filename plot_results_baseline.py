# -*- coding: utf-8 -*-
"""
@author: Abdul Hakmeh
@Email:ahak15@♣tu-claushtal.de
"""
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
import matplotlib.pyplot as plt
import numpy as np
import csv 
import os
from scipy.special import comb as n_over_k
import numpy as np
from scipy.special import comb
import pandas as pd
import matplotlib.font_manager as font_manager
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
import matplotlib.font_manager as font_manager

def call_result_baseline(functionValue_input,algo,func):   
    for functionValue in  functionValue_input: 
        full_vector={}
        full_vector_inverse={}
        full_vector_linear={}    
        for algo in algo:
            for spacing_function in func:
                for samplingFunction in ['no_function','inverse','max','min','median']:
                    for linear in ['False','True'] :
                        if samplingFunction == 'inverse' and linear=='True':
                            break                                             
                        if linear =='True':
                            if samplingFunction=='max':
                                full_vector_linear[algo,spacing_function,linear,samplingFunction,functionValue] = pd.read_csv('./results/'+'/'+algo+'/'+str(functionValue)+'/'+spacing_function+'/'+ algo +'_' + spacing_function +'_' +linear +'_' + samplingFunction + '_'+str(functionValue) +'.csv',index_col=False)[['F1_AV','history_aggregation_factor']]
                            else:
                                break                       
                        elif samplingFunction == 'inverse' :                                
                            full_vector_inverse[algo,spacing_function,linear,samplingFunction,functionValue] = pd.read_csv('./results/'+'/'+algo+'/'+str(functionValue)+'/'+spacing_function+'/'+ algo +'_' + spacing_function +'_' +linear +'_' + samplingFunction +'.csv',index_col=False)                            
                        else: 
                            full_vector[algo,spacing_function,linear,samplingFunction,functionValue] = pd.read_csv('./results/'+'/'+algo+'/'+str(functionValue)+'/'+spacing_function+'/'+ algo +'_' + spacing_function +'_' +linear +'_' + samplingFunction + '_'+str(functionValue) +'.csv',index_col=False)[['F1_AV','history_aggregation_factor']]
                                         
        csfont = {'fontname':'Times New Roman'}            
        fig, ax = plt.subplots()
        fig.set_size_inches(13, 8)
        markersize=8
        factor_=[]
        for factor in np.arange(0.001,0.15+0.002,0.002):
                  factor_.append(round(factor,3))
        fig.text(0.55, 0.15, 'History Aggregation Factor', ha='center', va='center', fontsize=30,**csfont)
        fig.text(0.017, 0.55, 'Averaged F1-Score', ha='center', va='center', rotation='vertical',fontsize=30,**csfont)                
        best_function='max'
        for element in full_vector:
          if element[3] == best_function:
              X=full_vector[element]
              X= X['history_aggregation_factor'].values
              if X[2]==1:
                  X=factor_
              X_=X    
              Ya=full_vector[element]
              Ya=Ya['F1_AV'].values
              ax.plot(X, Ya,marker='*',linewidth=2.5,c='y',label="quadratic function",markersize=markersize)
        for element in full_vector_linear:           
            if element[3] == best_function:
                X=full_vector_linear[element]
                X= X['history_aggregation_factor'].values
                if X[2]==1:
                    X=X_          
                Ya=full_vector_linear[element]
                Ya=Ya['F1_AV'].values           
                ax.plot(X,Ya,marker='v',label="linear function",c='b',linewidth=2.5,markersize=5)
        for element in full_vector_inverse:        
            X=full_vector_inverse[element]
            X= X['history_aggregation_factor'].values
            if X[2]==1:
                X=X_          
            Ya=full_vector_inverse[element]
            Ya=Ya['F1_AV'].values
            ax.plot(X, Ya,marker='x',alpha=0.6,label="inverse function",linewidth=2.5,c='r',markersize=markersize)    
            ax.margins(0.019)        
            sides=['top','bottom','left','right']
            for s in sides:
                ax.spines[s].set_linewidth(2)      
            font = font_manager.FontProperties(family="Times New Roman",style='normal', size=24)
       
            custom_lines =[ 
                        Line2D([0], [0], color='y', lw=3, marker='*',markersize=13),
                        Line2D([0], [0], color='b', lw=3, marker='v',markersize=12),
                        Line2D([0], [0], color='r', lw=3, marker="x",markersize=13)]
            ax.legend(custom_lines,['exponential function','Linear function','Inverse function'] ,prop=font,frameon=False,loc="upper center",bbox_to_anchor=(0.5, 1.117) ,ncol=3)            
            ax.set_yticklabels([f'{round(i,2):.2f}' for i in np.linspace(0.25,0.8,11,dtype=float)], fontsize=23,fontname = "Times New Roman")
            ax.set_xticklabels([f'{round(i,2):.2f}'  for i in np.linspace(0,max(X_),16,dtype=float)], fontsize=23,fontname = "Times New Roman")
            fig.subplots_adjust(top=0.87,
                bottom=0.227,
                left=0.095,
                right=0.975,
                hspace=0.416,
                wspace=0.2)
            label_Y=[f'{round(i,2):.2f}' for i in np.linspace(0.25,0.8,11,dtype=float)]           
            ax.yaxis.set_ticks([round(i,2) for i in np.linspace(0.25,0.8,11)])
            ax.xaxis.set_ticks([round(i,2) for i in np.linspace(0,max(X_),16)])
            ax.xaxis.set_ticklabels([])                        
            plt.savefig( './results/plots/Baselin Comparison/'+'baseline_'+algo+'_'+str(functionValue)+'_'+str(spacing_function)+'.pdf', format='pdf', dpi=1800)
###########################################################################
###########################################################################


algo=['s2s']
func=['quad']
val=[1.2]

call_result_baseline(val, algo, func)

