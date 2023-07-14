# -*- coding: utf-8 -*-
from pyT2FTS.T2FTS import Type2Model,IT2FS_plot,IT2FS_plot_OLD
from pyT2FTS.Tools import error_metrics,plot_forecast 
from pyT2FTS.Partitioners import SODA_part,ADP_part,ADP_part_antigo,DBSCAN_part,FCM_part,ENTROPY_part,CMEANS_part,HUARNG_part
from pyT2FTS.Transformations import Differential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
    Function that trains and tests a time series called 'data'

    :params:
    :data: data to be trained and tested
    :diff: flag to differentiate or not the input data
    :order: model order ( n. or lags in forecasting)
    :number_of_sets: n. of fuzzy sets created 
    
"""
    
def T2FTS(data,method_part,mf_type,partition_parameters,order,diff,training):
        
    '------------------------------------------------ Setup ------------------------------------------'
   
    """
    'codigo para rodar TAIEX POR ANO'
    'Separa por mes'
    g = data.groupby(pd.Grouper(key='Date', freq='M'))
    taiex_df_mes = [group for _,group in g]
    
    treino = taiex_df_mes[:10]
    teste = taiex_df_mes[10:]
    
    treino_df = pd.concat(treino)
    teste_df = pd.concat(teste)
    
    training_data = treino_df.Close.to_numpy() 
    test_data = teste_df.Close.to_numpy()   
    
    """
        
    if training == 1:   #(INFO PERFEITA)
        training_data = data
        test_data = data
    else:
        
        
   
        'Training takes % of the data'
        training_interval = int(training * len(data))     

        training_data = data[:training_interval]
        
        """
        #SE TIVER DADOS DE VALIDAÇÃO
        #obs: comentar a linha test_data = data[training_interval:]
        
        test = 0.15
        test_interval = int(test * len(data))  
        test_interval = test_interval * (-1)
        test_interval = test_interval - 1
        
        'Testing takes the remaining %'
        test_data = data[test_interval:]
        """
        

        'Testing takes the remaining %'
        test_data = data[training_interval:]
        
        
        
       
    'Checks if the data must be differentiated'
    if diff == True:
        training_data_orig = training_data
        test_data_orig = test_data
    
        tdiff = Differential(1) 
        training_data = tdiff.apply(training_data_orig)
        test_data = tdiff.apply(test_data_orig)
        
    
    'Create an object of the class Type2Model'
    modelo = Type2Model(training_data,order) 
    
    
        
    '------------------------------------------------ Fuzzy sets generation  -------------------------------------------------'

    if method_part == 'chen':
        number_of_sets = partition_parameters
        modelo.grid_partitioning(partition_parameters, mf_type)
        
    elif method_part == 'SODA':
        gridsize = partition_parameters
        number_of_sets = SODA_part(training_data,gridsize)
        modelo.grid_partitioning(number_of_sets, mf_type)   
        
    elif method_part == 'ADP':
        gridsize = partition_parameters
        number_of_sets,cloud_info = ADP_part(training_data, gridsize)
        #modelo.ADP_center_part_limites(cloud_info,number_of_sets, mf_type)
        modelo.grid_partitioning(number_of_sets, mf_type)    
     
 
    elif method_part == 'ADP_ANTIGO':
        gridsize = partition_parameters
        number_of_sets = ADP_part_antigo(training_data, gridsize)
        modelo.grid_partitioning(number_of_sets, mf_type)
    
    
    elif method_part == 'DBSCAN':
        eps = partition_parameters
        number_of_sets = DBSCAN_part(training_data, eps)
        modelo.grid_partitioning(number_of_sets, mf_type)
        
    elif method_part == 'CMEANS': 
        k = partition_parameters
        cmeans_params = CMEANS_part(training_data, k, mf_type)
        number_of_sets = len(cmeans_params)
        modelo.generate_uneven_length_mfs(number_of_sets, mf_type, cmeans_params)
    
    elif method_part == 'entropy':
        k = partition_parameters
        entropy_params = ENTROPY_part(training_data,k, mf_type)
        number_of_sets = len(entropy_params)
        modelo.generate_uneven_length_mfs(number_of_sets, mf_type, entropy_params)
        
    elif method_part == 'FCM':
        k = partition_parameters
        fcm_params = FCM_part(training_data,k, mf_type)
        number_of_sets = len(fcm_params)
        modelo.generate_uneven_length_mfs(number_of_sets, mf_type, fcm_params)
    
    elif method_part == 'huarng':
        huarng_params = HUARNG_part(training_data)
        number_of_sets = len(huarng_params)
        modelo.generate_uneven_length_mfs(number_of_sets,huarng_params)
        
        
    else:
        raise Exception("Method %s not implemented" % method_part)
        
        
    #Plot partition graphs
    #plot_title = str(number_of_sets) + ' partitions'
    #IT2FS_plot(*modelo.dict_sets.values(),title= plot_title,mf_shape = mf_type)
    
    
    
    '------------------------------------------------ Training  ------------------------------------------'
        
    'Treina o modelo'
    FLR,FLRG = modelo.training()
    
    '------------------------------------------------  Testing  ------------------------------------------'
    'Clips the test data for them to be inside the Universe of Discourse'
    test_data = np.clip(test_data, modelo.dominio_inf+1, modelo.dominio_sup-1)

    
    print("Partitioner:",method_part,"| N. of sets:", number_of_sets, "| Order:", order)
    print("")
    forecast_result = modelo.predict(test_data)   
    

    
    'Return values to original scale (i.e. undo the diff)'
    if diff == True:
        forecast_result = forecast_result[1:] #faz isso por causa da diferenciação
        forecast_result = tdiff.inverse(forecast_result,test_data_orig)
        proximo_valor_previsto = forecast_result[-1] #Esse é o proximo valor previsto da serie
        test_data = test_data_orig[order:]  # Para plotar e metricas de erro deve usar a serie original
        
    else:       
        test_data = test_data[order:]  # O primeiro item nao tem correspondente na previsao
        proximo_valor_previsto = forecast_result[-1] #Esse é o proximo valor previsto da serie
        forecast_result = forecast_result[:-1]
        

    '------------------------------------------------  Métricas de erro  ------------------------------------------'
    
    error_list = error_metrics(test_data,forecast_result)

        
    'Plots forecast graph data x forecast'      
    #plot_forecast(test_data,forecast_result)
    
    
    """
    para gerar grafico ADP-T2FTS com previsao de outros modelos
    name_file = "previsao_soda" + ".xlsx"      
    import pandas as pd
    writer = pd.ExcelWriter(name_file, engine='xlsxwriter')          
    fre = pd.Series(forecast_result)
    fre.to_excel(writer, sheet_name='General errors',index = False)
    writer.save()
    """           


    


    return error_list,number_of_sets,FLR,FLRG,proximo_valor_previsto


    """ 
    elif method_part == 'ADP':
        gridsize = partition_parameters
        number_of_sets,idx = ADP_part(training_data, gridsize)
        modelo.grid_partitioning(number_of_sets, mf_type) """
    




