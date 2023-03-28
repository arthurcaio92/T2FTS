
    
    # -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pyT2FTS.Headquarters import T2FTS
import pickle #To save the data as the process goes
import time

    
def run_Gridsearch(datasets,dataset_names,diff,partition_parameters,orders,partitioners,mfs,training):
    
    """
    
    Performs sliding window methodology over a time series
    
    params:
    :datasets: List of time series
    :dataset_names: List of time series names
    :diff: List of flag to differentiate or not the input data
    :partition_parameters: List of partition parameters to be used. If method_part is SODA or ADP, then partitions is the gridsize.
    :orders: List of List of model orders ( n. or lags in forecasting)
    :partitioners: List of partitioners to be used
    :mfs: List of membership functions to be used
    
    Return (currently disabled)
    
    :df_specific: dataframe with average errors accounting for all window  
    Saves an excel file (.xlsx) in the end with error metrics

    """
    
    'Checks if the right number and datasets and their names was provided'
    if len(datasets) != len(dataset_names):
        raise Exception("Please specify the correct number of datasets and their names")
        
    
    'Auxiliar variable to know the name of the data set'
    name_index = 0
    
    for data in datasets:
        
        data_name = dataset_names[name_index]
        name_index = name_index + 1 #update for the nex loop
        
        
        for method_part in partitioners:
            
            for mf_type in mfs:
           
                'Verifications'
                
                'If any of these situations happen, ignore this loop and goes to the next'
                
                if method_part == 'CMEANS' and (mf_type == 'trapezoidal' or mf_type == 'gaussian'):
                    print("-------------\n","WARNING: ", method_part," does not support ",mf_type," membership function","\n-------------",)
                    continue
                
                if (method_part == 'FCM' or method_part == 'entropy') and (mf_type == 'gaussian'):
                    print(method_part," does not support ",mf_type," membership function")
                    continue
                
                'Let''s measure the total elapsed time for the whole process to be completed'
                start_time = time.time()
    
            
                'list to sabe the errors'
                lista_rmse = []
                lista_partitions = []
                lista_rules = []
                lista_flrg = []
                
 
                specific_errors = {'Gridsize':[],
                                    'Order':[],
                                    'Partitions': [],
                                    'FLR':[],
                                    'FLRG':[],
                                    'RMSE':[],
                                    'MAPE':[],
                                    'MAE':[],
                                    'Time(s)': [],
                                    'Total Time(s)': []
                          
                                     }  
                
                
                '-----Begins the Gridsearch------'
                
                for part_param in partition_parameters:
                    
                    gridsize = part_param
                    
                    for lag in orders:
                    
                                               
                        'Let''s measure the METHOD elapsed time '
                        method_start_time = time.time()
                        
                
                        'Define model order'
                        order = lag
                        
                                        
                        lista_erros,n_sets,FLR,FLRG = T2FTS(data,method_part,mf_type,part_param,order=order,diff=diff,training=training)
                       
                        print("---------------------------------")
                           
                        '------------------------------------------------  Error Metrics  ------------------------------------------'
                        'Gets the RMSE from the errors list'
                        rmse = lista_erros[3]
                        mape = lista_erros[1]
                        mae = lista_erros[4]
                        
                        'Adds the number of rules to the respective list'
                        lista_rules.append(FLR)
                        lista_flrg.append(FLRG)
                       
                        
                        'Ends time measurement'
                        method_end_time = time.time()            
                        method_elapsed_time = method_end_time - method_start_time
                        
                        
                        'Builds the specific_error dictionary'   
                                              
                        specific_errors['Gridsize'].append(gridsize)   
                        specific_errors['Order'].append(order)         
                        specific_errors['Partitions'].append(n_sets) 
                        specific_errors['FLR'].append(FLR)  
                        specific_errors['FLRG'].append(FLRG)  
                        specific_errors['RMSE'].append(rmse)
                        specific_errors['MAPE'].append(mape)
                        specific_errors['MAE'].append(mae)
                        specific_errors['Time(s)'].append(method_elapsed_time)
                        specific_errors['Total Time(s)'].append(None)
            
                        
                        'Prints the results'
                        if method_part == 'chen':
                            r = "RMSE avg - part: " + str(part_param) + ", Order: " + str(order)
                            print("[",r,"]:",rmse)
                            print("---------------------------------")
            
                        elif method_part == 'SODA' or method_part == 'ADP': 
                            r = "RMSE avg - Gridsize: " + str(gridsize) + ", Order: " + str(order)
                            print("[",r,"]:",rmse)
                            print("---------------------------------")
                        
                        else:
                            r = "RMSE avg - Parâmetro: " + str(gridsize) + ", Order: " + str(order)
                            print("[",r,"]:",rmse)
                            print("---------------------------------")
                        
                            
                                        
                        'Use pickle to save the dicts as backup'
                        """
                        pickle_out = open("specific.pickle","wb")          
                        pickle.dump(specific_errors, pickle_out)         
                        pickle_out.close()
                        """
                        
                        'Resets the lists'
                        lista_rmse = []  
                        lista_rules = []
                        lista_flrg = []
                        lista_partitions = []
                        
                
                'Ends time measurement'
                end_time = time.time()      
                total_elapsed_time = end_time - start_time
                
                'Adds the final line with the total elapsed time'
                specific_errors['Gridsize'].append(None)   
                specific_errors['Order'].append(None)         
                specific_errors['Partitions'].append(None) 
                specific_errors['FLR'].append(None)  
                specific_errors['FLRG'].append(None)  
                specific_errors['RMSE'].append(None) 
                specific_errors['MAPE'].append(None) 
                specific_errors['MAE'].append(None) 
                specific_errors['Time(s)'].append('Total Elapsed Time:')
                specific_errors['Total Time(s)'].append(total_elapsed_time)
                           
                
                '------------------------------------------------  Save to excel  ------------------------------------------'
            
                'Defines the name of the final file'
                if diff == 0:  
                    name_file = method_part + "_semdiff_" + data_name + "_" + mf_type + "_" + str(partition_parameters[0]) + "a" + str(partition_parameters[-1]) + ".xlsx"
                    
                elif diff == 1:   
                    name_file = method_part + "_diff_" + data_name + "_" + mf_type + "_" + str(partition_parameters[0]) + "a" + str(partition_parameters[-1]) + ".xlsx"      
                       
                
                print("Saved file:",name_file)
                writer = pd.ExcelWriter(name_file, engine='xlsxwriter')
                                        
                df_specific = pd.DataFrame(data=specific_errors)
                #df_especifico.columns = ['Gridsize','Partitions','RMSE medio 1_Order', 'Desvio padrao RMSE 1_Order','RMSE medio 2_Order', 'Desvio padrao RMSE 2_Order','RMSE medio 3_Order', 'Desvio padrao RMSE 3_Order','FLR','FLRG']    
                       
                df_specific.to_excel(writer, sheet_name='Especific errors',index = False)
               
                writer.save()
                
                #Downloads the Excel file to computer
                #from google.colab import files
                #files.download(name_file)
                
                
             
                
        
        

