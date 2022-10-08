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

#######################################################bizar interpolation######################
def get_bezier_parameters(X, Y, degree=3):
    """ Least square qbezier fit using penrose pseudoinverse.

    Parameters:

    X: array of x data.
    Y: array of y data. Y[0] is the y point for X[0].
    degree: degree of the Bézier curve. 2 for quadratic, 3 for cubic.

    Based on https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
    and probably on the 1998 thesis by Tim Andrew Pastva, "Bézier Curve Fitting".
    """
    if degree < 1:
        raise ValueError('degree must be 1 or greater.')

    if len(X) != len(Y):
        raise ValueError('X and Y must be of the same length.')

    if len(X) < degree + 1:
        raise ValueError(f'There must be at least {degree + 1} points to '
                         f'determine the parameters of a degree {degree} curve. '
                         f'Got only {len(X)} points.')

    def bpoly(n, t, k):
        """ Bernstein polynomial when a = 0 and b = 1. """
        return t ** k * (1 - t) ** (n - k) * comb(n, k)
        #return comb(n, i) * ( t**(n-i) ) * (1 - t)**i
    def bmatrix(T):
        """ Bernstein matrix for Bézier curves. """
        return np.matrix([[bpoly(degree, t, k) for k in range(degree + 1)] for t in T])
    def least_square_fit(points, M):
        M_ = np.linalg.pinv(M)
        return M_ * points
    T = np.linspace(0, 1, len(X))
    M = bmatrix(T)
    points = np.array(list(zip(X, Y))) 
    final = least_square_fit(points, M).tolist()
    final[0] = [X[0], Y[0]]
    final[len(final)-1] = [X[len(X)-1], Y[len(Y)-1]]
    return final
def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i
def bezier_curve(points, nTimes=50):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])
    t = np.linspace(0.0, 1.0, nTimes)
    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    return xvals, yvals
def call_result_approximation(functionValue_input,algo,function):        
    for functionValue in  functionValue_input: 
        full_vector={}
        full_vector_inverse={}
        full_vector_linear={}    
        for algo in algo:
            for spacing_function in function:
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
        degree=5
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
              data = get_bezier_parameters(X, Ya, degree=degree)
              x_val = [x[0] for x in data]
              y_val = [x[1] for x in data]
              xvals, yvals = bezier_curve(data, nTimes=1000)
              ax.plot(xvals, yvals,linewidth=2.5,c='y')
              ax.plot(X,Ya,'*',label="quadratic function",c='y',markersize=markersize)
        for element in full_vector_linear:           
            if element[3] == best_function:
                X=full_vector_linear[element]
                X= X['history_aggregation_factor'].values
                if X[2]==1:
                    X=X_
          
                Ya=full_vector_linear[element]
                Ya=Ya['F1_AV'].values             
                data = get_bezier_parameters(X, Ya, degree=degree)
                x_val = [x[0] for x in data]
                y_val = [x[1] for x in data]
                xvals, yvals = bezier_curve(data, nTimes=1000)
                ax.plot(xvals,yvals,label="linear function",c='b',linewidth=2.5)     
                ax.plot(X,Ya,"v",c='b',markersize=5)
            
        for element in full_vector_inverse:
   
            X=full_vector_inverse[element]
            X= X['history_aggregation_factor'].values
            if X[2]==1:
                X=X_         
            Ya=full_vector_inverse[element]
            Ya=Ya['F1_AV'].values
            data = get_bezier_parameters(X, Ya, degree=degree)
            x_val = [x[0] for x in data]
            y_val = [x[1] for x in data]
            xvals, yvals = bezier_curve(data, nTimes=1000)
            ax.plot(xvals, yvals,alpha=0.6,label="inverse function",linewidth=2.5,c='r')
            ax.plot(X,Ya,'x',c='r',markersize=markersize)           
            ax.margins(0.019)                             
            sides=['top','bottom','left','right']
            for s in sides:
                ax.spines[s].set_linewidth(2)                  
            font = font_manager.FontProperties(family="Times New Roman",style='normal', size=24)
       
            custom_lines =[ 
                        Line2D([0], [0], color='y', lw=3, marker='*',markersize=13),
                        Line2D([0], [0], color='b', lw=3, marker='v',markersize=12),
                        Line2D([0], [0], color='r', lw=3, marker="x",markersize=13)            
                        ]
            ax.legend(custom_lines,['quadratic function','Linear function','Inverse function'] ,prop=font,frameon=False,loc="upper center",bbox_to_anchor=(0.5, 1.117) ,ncol=3)           
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
            plt.savefig( './results/plots/Aproximation/'+'approx'+'_'+algo+'_'+str(functionValue)+'_'+str(spacing_function)+'.pdf', format='pdf', dpi=1800)
        
        
#######################################################################        
#######################################################################
algo=['s2s']
func=['quad']
val=[1.1]

call_result_approximation(val, algo, func)

