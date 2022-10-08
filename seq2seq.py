import random as python_random
import numpy as np
import tensorflow as tf
np.random.seed(123)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(123)
# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(1234)
from __future__ import print_function, division
from warnings import warn
from nilmtk.disaggregate import Disaggregator
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Reshape, Flatten
import os
import pandas as pd
import pickle
from collections import OrderedDict
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential, load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from scipy.signal import get_window
from scipy import signal
class SequenceLengthError(Exception):
    pass
class ApplianceNotFoundError(Exception):
    pass
class Seq2Seq(Disaggregator):
    def avg_of_top_n(self, l,n):
      return sum(sorted(l)[-n:]) / n
    def __init__(self, params):
        self.MODEL_NAME = "Seq2Seq"
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',561)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = OrderedDict()
        self.mains_mean = 1800
        self.mains_std = 600
        self.batch_size = params.get('batch_size',512)
        self.appliance_params = params.get('appliance_params',{})
        if self.sequence_length%2==0:
            print ("Sequence length should be odd!")
            raise (SequenceLengthError)
        self.useWindow= params.get('window', 'none' )
        self.aggregation_factor = params.get('history_aggregation_factor', 0.05 )
        self.windowing=  params.get('use_Hakmeh_windowing', False )
        self.spacing_function_value   =params.get('time_spacing_function_value', 100 )
        self.spacing_function = params.get('spacing_function','quad')
        self.sampling_function= params.get('sampling_function','max')
        self.linear_function= params.get('linear_function', False)
    def partial_fit(self,train_main,train_appliances,do_preprocessing=True,**load_kwargs):

        print("...............Seq2Seq partial_fit running...............")
        print("*"*7,'Expriment parameters',"*"*7)
        print("history aggregation factor= ",self.aggregation_factor)       
        print("Total sequance length= ",self.sequence_length)      
        print("equally spaced length= ", round(self.sequence_length- (self.aggregation_factor *self.sequence_length)))
        print("nonlinear part length= ", round(self.aggregation_factor *self.sequence_length))
        print("spacing function value= " , self.spacing_function_value)      
        print("spacing function= " ,self.spacing_function)
        print("sampling function= " ,self.sampling_function)
        print("use linear as spacing function= " ,self.linear_function)    
        print("Batch size= ", self.batch_size)
        print("*"*20)
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)
        print (len(train_main))
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')
        train_main = pd.concat(train_main,axis=0)
        train_main = train_main.values.reshape((-1,self.sequence_length,1))       
        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df = pd.concat(app_dfs,axis=0)
            app_df_values = app_df.values.reshape((-1,self.sequence_length))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances

        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print("First model training for ", appliance_name)
                self.models[appliance_name] = self.return_network()
            else:
                print("Started Retraining model for ", appliance_name)
            model = self.models[appliance_name]
            if train_main.size > 0:
                # Sometimes chunks can be empty after dropping NANS
                if len(train_main) > 10:
                    # Do validation when you have sufficient samples
                    filepath = './weights/s2s/seq2seq-temp-weights-'+str(python_random.randint(0,100000))+'.h5'
                    checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
                    train_x, v_x, train_y, v_y = train_test_split(train_main, power, test_size=.15,random_state=10)
                    model.fit(train_x,train_y,validation_data=(v_x,v_y),epochs=self.n_epochs,callbacks=[checkpoint],batch_size=self.batch_size)
                    model.load_weights(filepath)                   
    def disaggregate_chunk(self,test_main_list,model=None,do_preprocessing=True):

        if model is not None:
            self.models = model
        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, submeters_lst=None, method='test')
        test_predictions = []
        for test_mains_df in test_main_list:
            disggregation_dict = {}
            test_main_array = test_mains_df.values.reshape((-1, self.sequence_length, 1))
            for appliance in self.models:
                prediction = []
                model = self.models[appliance]
                prediction = model.predict(test_main_array ,batch_size=self.batch_size)
                #####################
                # This block is for creating the average of predictions over the different sequences
                # the counts_arr keeps the number of times a particular timestamp has occured
                # the sum_arr keeps the number of times a particular timestamp has occured
                # the predictions are summed for  agiven time, and is divided by the number of times it has occured               
                l = self.sequence_length
                n = len(prediction) + l - 1
                sum_arr = np.zeros((n))
                counts_arr = np.zeros((n))
                o = len(sum_arr)
                for i in range(len(prediction)):
                    sum_arr[i:i + l] += prediction[i].flatten()
                    counts_arr[i:i + l] += 1
                for i in range(len(sum_arr)):
                    sum_arr[i] = sum_arr[i] / counts_arr[i]
                #################
                prediction = self.appliance_params[appliance]['mean'] + (sum_arr * self.appliance_params[appliance]['std'])
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    def return_network(self):
        model = Sequential()
        # 1D Conv
        model.add(Conv1D(30,10,activation="relu",input_shape=(self.sequence_length,1),strides=2))
        model.add(Conv1D(30, 8, activation='relu', strides=2))
        model.add(Conv1D(40, 6, activation='relu', strides=1))
        model.add(Conv1D(50, 5, activation='relu', strides=1))
        model.add(Dropout(.2))
        model.add(Conv1D(50, 5, activation='relu', strides=1))
        model.add(Dropout(.2))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(self.sequence_length))
        model.compile(loss='mse', optimizer='adam')
        return model   
    def call_preprocessing(self, mains_lst, submeters_lst, method):
        if method == 'train':            
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                linear_length = round(n- (self.aggregation_factor * n))
                nonlinear_sample_length = n- linear_length
                new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0)) 
          
                #########################Hakmeh windowing#######################################
                if self.windowing : 
                    print("*"*20)
                    print('using Hakmeh Windowing')
                    print("*"*20)
                    for app_index, (app_name, app_df_list) in enumerate(submeters_lst):
                        for app_df in app_df_list:
                            new_app_readings = app_df.values.reshape((-1, 1))                   
                    coverd_time=0  
                    for i in range(1,nonlinear_sample_length+1):
                        if self.spacing_function == 'expo':  
                            coverd_time= coverd_time + pow(self.spacing_function_value,i)
                        elif  self.spacing_function == 'quad':     
                           coverd_time= coverd_time +  pow(i,self.spacing_function_value)
                    coverd_time=round(coverd_time)
                    linear_stride=coverd_time//nonlinear_sample_length                  
                    new_mains_=[]
                    for j in range(len(new_mains)-n +1 ):
                        sequance_index= np.linspace(j, j + n  ,n, dtype=int, endpoint=False)                                                           
                        last_sequnce= sequance_index[ nonlinear_sample_length-1 :].tolist()                                         
                        temp= last_sequnce[0]
                        if temp < coverd_time :                          
                            mains_=new_mains[0:temp] 
                        else:                            
                            mains_=new_mains[temp-coverd_time:temp]                             
                        mains_=mains_[::-1]                 
                        if self.aggregation_factor < 1 and self.sampling_function!='no_function' :
                            step=0                                          
                            max_=[]
                            for i in range(2,nonlinear_sample_length+1):                             
                                if (self.spacing_function == 'quad' or  self.spacing_function == 'quad') and self.linear_function :                                    
                                    indx= linear_stride + step                             
                                if self.spacing_function == 'expo':
                                    indx=round(pow(self.spacing_function_value,i)) + step                                   
                                elif self.spacing_function == 'quad':
                                    indx=round(pow(i,self.spacing_function_value)) + step                                   
                                if indx > len(mains_):
                                    break
                                if self.sampling_function=='max':
                                    max_.append(max(mains_[step:indx]))
                                elif self.sampling_function=='average':
                                    max_.append(np.average(mains_[step:indx]))
                                elif self.sampling_function=='median':                                   
                                    max_.append(np.median(mains_[step:indx]))
                                elif self.sampling_function=='min':                                   
                                    max_.append(min(mains_[step:indx]))
                                step= indx
                            max_=np.flip(max_).tolist()
                            last_sequnce_= new_mains[last_sequnce].tolist()
                            last_sequnce_= max_+ last_sequnce_      
                            if len(last_sequnce_)< n: 
                                last_sequnce_=np.pad(last_sequnce_, (n - len(last_sequnce_),0) ,'constant',constant_values = (0,0))                           
                            new_mains_.extend(last_sequnce_[-n:])                      
                        else:                           
                            for k in range(2,nonlinear_sample_length +1  ): # fill the middle sequance to full seuqnace with expo. downsampling                             
                                if self.linear_function:
                                    backward_index= last_sequnce[0] - linear_stride 
                                elif self.spacing_function == 'quad':
                                    backward_index=  last_sequnce[0] - round(pow(k,self.spacing_function_value))
                                elif self.spacing_function == 'expo':    
                                    backward_index=  last_sequnce[0] - round(pow(self.spacing_function_value,k))                              
                                if backward_index >= 0:
                                    last_sequnce = [backward_index] + last_sequnce # push at the end                               
                                else:                                      
                                     last_sequnce= [0] +last_sequnce
                                                       
                            new_mains_.extend(new_mains[last_sequnce])                          
                    new_mains= np.array([new_mains_[i:i + n] for i in range(0,len(new_mains_),n)])                   
                    new_mains = (new_mains - self.mains_mean) / self.mains_std                                                 
                else:                    
                   if self.useWindow!='none':
                         window= get_window(self.useWindow, n)
                         print("window= ",self.useWindow)
                         w_half=window[0:(n//2)+1]
                         x= np.linspace(0,(n//2)+1,(n//2)+1,dtype=int)
                         x_new =np.linspace(0,(n//2)+1,n)
                         w_half=np.interp(x_new,x,w_half)
                         for i in range(len(new_mains)):
                               new_mains[i]= w_half *   new_mains[i]                               
                   new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                   new_mains = (new_mains - self.mains_mean) / self.mains_std
                processed_mains_lst.append(pd.DataFrame(new_mains))
            ################################################################################
            appliance_list = []
            for app_index, (app_name, app_df_lst) in enumerate(submeters_lst):
                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                else:
                    print ("Parameters for ", app_name ," were not found!")
                    raise ApplianceNotFoundError()
                processed_app_dfs = []
                for app_df in app_df_lst:                    
                    new_app_readings = app_df.values.flatten()
                    new_app_readings = np.pad(new_app_readings, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                    new_app_readings = np.array([new_app_readings[i:i + n] for i in range(len(new_app_readings) - n + 1)])                    
                    new_app_readings = (new_app_readings - app_mean) / app_std  # /self.max_val
                    processed_app_dfs.append(pd.DataFrame(new_app_readings))                                      
                appliance_list.append((app_name, processed_app_dfs))
            return processed_mains_lst, appliance_list
        else:
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                #new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                new_mains = new_mains.reshape((-1, self.sequence_length))
                processed_mains_lst.append(pd.DataFrame(new_mains))
            return processed_mains_lst
    def set_appliance_params(self,train_appliances):

        for (app_name,df_list) in train_appliances:
            l = np.array(pd.concat(df_list,axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std<1:
                app_std = 100
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})
            

