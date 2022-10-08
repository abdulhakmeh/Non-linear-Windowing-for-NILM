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

def call_result_samplingFunctions(functionValue_input,algo,func):  
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
        fig, axs = plt.subplots(len(full_vector), 1)
        fig.set_size_inches(13, 11)
        factor_=[]
        for factor in np.arange(0.001,0.15+0.002,0.002):
                  factor_.append(round(factor,3))
        fig.text(0.5, 0.02, 'History Aggregation Factor', ha='center', va='center', fontsize=30,**csfont)
        fig.text(0.019, 0.5, 'Averaged F1-Score', ha='center', va='center', rotation='vertical',fontsize=30,**csfont) 
        font = font_manager.FontProperties(family="Times New Roman",style='normal', size=22)
        font_16 = font_manager.FontProperties(family="Times New Roman",style='normal', size=22)
        plt.show()
        index=0
        color=['b','r','black','g','c']
        labels=['No function','Max','Min','Median','Average']
        for element_ in full_vector:
            index_color=0
            for element in full_vector:
                X=full_vector[element]
                X= X['history_aggregation_factor'].values
                if X[0]==1:
                    X=factor_              
                Ya=full_vector[element]
                Ya=Ya['F1_AV'].values
                if element[3]==element_[3]:
                     axs[index].plot(X, Ya, c=color[index_color],linewidth=2)                    
                     xmax= X[np.argmax(Ya)]
                     ymax = Ya.max()
                     axs[index].plot(xmax, ymax,marker='*', ms = 15,color=color[index_color])
                     xmax = X[np.argmax(Ya)]
                     ymax = Ya.max()
                     if not axs[index]:
                         ax=plt.gca()                        
                     custom_lines =[ Line2D([0], [0],markerfacecolor=color[index_color],markeredgecolor=color[index_color],lw=1,linestyle='',marker='*',markersize=13)]
                     axs[index].legend(custom_lines,['F1-Score= '+str(round(ymax,3)) ] ,prop=font,handletextpad=-0.1,frameon=False,loc="lower left",bbox_to_anchor=(-0.018, -0.1))                                  
                else:                    
                     axs[index].plot(X, Ya,alpha=0.23,c=color[index_color],linewidth=2)
                index_color+=1          
            index+=1
        for i in range(len(axs)):
                 axs[i].margins(0.01)          
        for i in range(len(axs)):
                axs[i].yaxis.set_ticks([round(i,2) for i in np.linspace(0.5,0.9,5)])
                axs[i].set_yticklabels([round(i,2) for i in np.linspace(0.5,0.9,5)], fontsize=20,fontname = "Times New Roman")
                if i != len(axs)-1:
                    axs[i].tick_params(labelbottom=False)                  
                axs[i].set_xticklabels([round(i,2) for i in np.linspace(0,max(X),16,endpoint=True)], fontsize=20,fontname = "Times New Roman")
                fig = axs[i].get_figure()
                fig.tight_layout()
                sides=['top','bottom','left','right']     
                for s in sides:
                    axs[i].spines[s].set_linewidth(1)      
        custom_lines =[Line2D([0], [0],color=color[i],lw=4) for i in range(len(full_vector))]                     
        leg= Legend(axs[0],custom_lines,[label for label in labels] ,handletextpad=0.9,prop=font_16,frameon=False,loc="upper center",bbox_to_anchor=(0.5, 1.31) ,ncol=5)
        axs[0].add_artist(leg)       
        axs[i].set_xticklabels([round(i,2) for i in np.linspace(0,max(X),16,endpoint=True)], fontsize=20,fontname = "Times New Roman")
        for i in range(len(axs)):  
            axs[i].xaxis.set_ticks([round(i,2) for i in np.linspace(0,max(X),16,endpoint=True)])                         
        fig.subplots_adjust(top=0.962,bottom=0.076,left=0.075,right=0.97,hspace=0.19,wspace=0.2)        
        plt.savefig( './results/plots/Sampling functions/'+'sam_fun'+'_'+algo+'_'+str(functionValue)+'_'+str(spacing_function)+'.pdf', format='pdf', dpi=1800) 
        
#######################################################################
#######################################################################


algo=['s2s']
func=['quad']
val=[1.2]

call_result_samplingFunctions(val, algo, func)(val, algo, func)

