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
from warnings import warn
from nilmtk.disaggregate import Disaggregator
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Reshape, Flatten
import os
import pickle
import pandas as pd
from collections import OrderedDict
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential, load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
import random
import sys
from scipy.signal import get_window
from scipy import signal
class SequenceLengthError(Exception):
    pass
class ApplianceNotFoundError(Exception):
    pass
class Seq2Point(Disaggregator):
    def __init__(self, params):
        """
        Parameters to be specified for the model
        """
        self.MODEL_NAME = "Seq2Point"
        self.models = OrderedDict()
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',99)
        self.n_epochs = params.get('n_epochs', 10 )
        self.batch_size = params.get('batch_size',512)
        self.appliance_params = params.get('appliance_params',{})
        self.mains_mean = params.get('mains_mean',1800)
        self.mains_std = params.get('mains_std',600)
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
    def partial_fit(self,train_main,train_appliances,do_preprocessing=True,
            **load_kwargs):
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)
        print("...............Seq2Point partial_fit running...............")
        # Do the pre-processing, such as  windowing and normalizing
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
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')
        train_main = pd.concat(train_main,axis=0)
        train_main = train_main.values.reshape((-1,self.sequence_length,1))       
        new_train_appliances = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df,axis=0)
            app_df_values = app_df.values.reshape((-1,1))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances
        for appliance_name, power in train_appliances:
            # Check if the appliance was already trained. If not then create a new model for it
            if appliance_name not in self.models:
                print("First model training for ", appliance_name)
                self.models[appliance_name] = self.return_network()
            # Retrain the particular appliance
            else:
                print("Started Retraining model for ", appliance_name)
            model = self.models[appliance_name]
            if train_main.size > 0:
                # Sometimes chunks can be empty after dropping NANS
                if len(train_main) > 10:
                    # Do validation when you have sufficient samples
                    filepath = './weights/s2p/seq2pint-temp-weights-'+str(random.randint(0,100000))+'.h5'
                    checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
                    train_x, v_x, train_y, v_y = train_test_split(train_main, power, test_size=.15,random_state=10)
                    model.fit(train_x,train_y,validation_data=(v_x,v_y),epochs=self.n_epochs,callbacks=[checkpoint],batch_size=self.batch_size)
                    model.load_weights(filepath)
    def disaggregate_chunk(self,test_main_list,model=None,do_preprocessing=True):
        if model is not None:
            self.models = model
        # Preprocess the test mains such as windowing and normalizing
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None, method='test')
        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values
            test_main = test_main.reshape((-1, self.sequence_length, 1))
            disggregation_dict = {}
            for appliance in self.models:
                prediction = self.models[appliance].predict(test_main,batch_size=self.batch_size)
                prediction = self.appliance_params[appliance]['mean'] + prediction * self.appliance_params[appliance]['std']
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions
    def return_network(self):
        # Model architecture
        model = Sequential()
        model.add(Conv1D(30,10,activation="relu",input_shape=(self.sequence_length,1),strides=1))
        model.add(Conv1D(30, 8, activation='relu', strides=1))
        model.add(Conv1D(40, 6, activation='relu', strides=1))
        model.add(Conv1D(50, 5, activation='relu', strides=1))
        model.add(Dropout(.2))
        model.add(Conv1D(50, 5, activation='relu', strides=1))
        model.add(Dropout(.2))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')  # ,metrics=[self.mse])
        return model
    def call_preprocessing(self, mains_lst, submeters_lst, method):
        if method == 'train':
            # Preprocessing for the train data
            mains_df_list = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                linear_length = round(n- (self.aggregation_factor * n))
                nonlinear_sample_length = n- linear_length
                new_mains = np.pad(new_mains,(units_to_pad,units_to_pad),'constant',constant_values=(0,0))
                #########################################Hakmeh windowing####################################################             
                if self.windowing : 
                    print("*"*20)
                    print('using Hakmeh Windowing')
                    print("*"*20)                  
                    coverd_time=0  
                    for i in range(nonlinear_sample_length):
                        if self.spacing_function == 'expo':  
                            coverd_time= coverd_time + pow(self.spacing_function_value,i)
                        elif  self.spacing_function == 'quad':     
                           coverd_time= coverd_time +  pow(i,self.spacing_function_value)
                    coverd_time=round(coverd_time)
                    liear_stride=coverd_time//nonlinear_sample_length
                    if liear_stride < 1 :
                      liear_stride+=1                    
                    new_mains_=[]                  
                    if self.aggregation_factor < 1 and self.sampling_function!='no_function' :
                        for j in range(len(new_mains)-n +1 ):
                            sequance_index= np.linspace(j, j + n  ,n, dtype=int, endpoint=False)                                           
                            # end_ind=sequance_index[-1]
                            last_sequnce= sequance_index[ nonlinear_sample_length-1 :].tolist()                                         
                            temp= last_sequnce[0]
                            if temp < coverd_time :
                                
                                mains_=new_mains[0:temp] 
                            else:
                                
                                mains_=new_mains[temp-coverd_time:temp] 
                                
                            mains_=mains_[::-1]
                            step=0                                            
                            max_=[]
                            for i in range(1,nonlinear_sample_length):
                                
                                if (self.spacing_function == 'expo' or  self.spacing_function == 'quad') and self.linear_function :                                    
                                    indx= liear_stride + step                            
                                elif self.spacing_function == 'expo':
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
                        mains_f = np.array([new_mains_[i:i + n] for i in range(0,len(new_mains_),n)])
                        mains_f = (mains_f - self.mains_mean) / self.mains_std                            
                    else:                        
                        for j in range(len(new_mains)-n +1 ):
                            sequance_index= np.linspace(j, j + n  ,n, dtype=int, endpoint=False)                                            
                            # end_ind=sequance_index[-1]
                            last_sequnce= sequance_index[ nonlinear_sample_length-1 :].tolist()                                                                                                                                                           
                            for k in range(1,nonlinear_sample_length): # fill the middle sequance to full seuqnace with expo. downsampling                             
                                if self.linear_function:
                                    backward_index= last_sequnce[0] - liear_stride 
                                elif self.spacing_function == 'quad':
                                    backward_index=  last_sequnce[0] - round(pow(k,self.spacing_function_value))
                                elif self.spacing_function == 'expo':    
                                    backward_index=  last_sequnce[0] - round(pow(self.spacing_function_value,k))                              
                                if backward_index >= 0:
                                    last_sequnce = [backward_index] + last_sequnce # push at the end                               
                                else:                                      
                                     last_sequnce= [0] +last_sequnce                                                       
                            new_mains_.extend(new_mains[last_sequnce])                           
                        mains_f= np.array([new_mains_[i:i + n] for i in range(0,len(new_mains_),n)])                   
                        mains_f = (mains_f - self.mains_mean) / self.mains_std                  
                else:                      
                    if self.useWindow!='none':
                          window= get_window(self.useWindow, n)
                          print("window= ",self.useWindow)
                          w_half=window[0:(n//2)+1]
                          x= np.linspace(0,(n//2)+1,(n//2)+1,dtype=int)
                          x_new =np.linspace(0,(n//2)+1,n)
                          w_half=np.interp(x_new,x,w_half)
                          for i in range(len(new_mains)):
                                new_mains[i]= w_half * new_mains[i]
                          mains_f = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                          mains_f = (mains_f - self.mains_mean) / self.mains_std
                    else:              
                         mains_f = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                         mains_f = (mains_f - self.mains_mean) / self.mains_std                                            
                mains_df_list.append(pd.DataFrame(mains_f))
                
                ############################################################################################################
            appliance_list = []
            for app_index, (app_name, app_df_list) in enumerate(submeters_lst):
                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                else:
                    print ("Parameters for ", app_name ," were not found!")
                    raise ApplianceNotFoundError()
                processed_appliance_dfs = []
                for app_df in app_df_list:
                    new_app_readings = app_df.values.reshape((-1, 1))
                    # This is for choosing windows
                    new_app_readings = (new_app_readings - app_mean) / app_std  
                    # Return as a list of dataframe
                    processed_appliance_dfs.append(pd.DataFrame(new_app_readings))
                appliance_list.append((app_name, processed_appliance_dfs))
            return mains_df_list, appliance_list
        else:  
            mains_df_list = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains,(units_to_pad,units_to_pad),'constant',constant_values=(0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))
            return mains_df_list
    def set_appliance_params(self,train_appliances):
        # Find the parameters using the first
        for (app_name,df_list) in train_appliances:
            l = np.array(pd.concat(df_list,axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std<1:
                app_std = 100
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})
        print (self.appliance_params)
