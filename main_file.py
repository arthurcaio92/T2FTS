# -*- coding: utf-8 -*-
from pyT2FTS.sliding_window import run_sliding_window
from pyT2FTS.datasets import get_TAIEX,get_NASDAQ,get_Brent_Oil,get_SP500
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'------------------------------------------------ Data set import -------------------------------------------------'
"""
taiex_df = get_TAIEX()
taiex = taiex_df.avg               
taiex = taiex.to_numpy()  

nasdaq_df = get_NASDAQ()
nasdaq = nasdaq_df.avg               
nasdaq = nasdaq.to_numpy()    

sp500_df = get_SP500()
sp500 = sp500_df.Avg               
sp500 = sp500[11500:16000]
sp500 = sp500.to_numpy()     

df_brent_oil = get_Brent_Oil()
brent_oil = df_brent_oil.Price  
brent_oil = brent_oil.to_numpy()

crude_oil_df = pd.read_excel('Crude-oil-prices.xls', sheet_name='Data 1')
crude_oil_df = crude_oil_df['RWTC']
crude_oil = crude_oil_df[1:].to_numpy()
crude_oil = crude_oil[5524:]

'n sei pq mas precisa fazer isso'
a=[]
for x in crude_oil:
    a.append(x)
    
crude_oil  = np.array(a)

 

taiex_df = get_TAIEX()
taiex = taiex_df.avg               
taiex = taiex.to_numpy() 
"""
nasdaq_df = get_NASDAQ()
nasdaq = nasdaq_df.avg               
nasdaq = nasdaq.to_numpy()  


'------------------------------------------------ Gridsearch Parameters -------------------------------------------------'


datasets = [nasdaq]
dataset_names = ['nasdaq']
diff = 1                                #If diff = 1, data is differentiated
partition_parameters = np.arange(5,6)            #partiions must be a list
orders = [1]
partitioners = ['SODA']                 #partitioners: 'chen' 'SODA' 'ADP' 'DBSCAN' 'CMEANS' 'entropy' 'FCM'  
mfs = ['triangular']                    #mfs: 'triangular' ou 'trapezoidal' ou 'gaussian'


'------------------------------------------------ Running the model -------------------------------------------------'


'Builds and runs the model'
#run_sliding_window(datasets,dataset_names,diff,partition_parameters,orders,partitioners,mfs,training = 0.8)






"############ RUN WITHOUT SLIDING WINDOW ############"

from Gridsearch import run_Gridsearch



"""
taiex_df = get_TAIEX()
taiex = taiex_df.avg               
taiex = taiex.to_numpy() 


'##### ENROLLMENTS #####'
enroll = pd.read_excel('Enrollments.xlsx')
ano = enroll.Year
enroll = enroll.Enrollments.to_numpy()


'##### MILHO #####'
enroll = pd.read_excel('wheat_production.xlsx')
ano = enroll.Year
enroll = enroll['Actual production (kg/ha)']
enroll = enroll.to_numpy()

taiex_df = get_TAIEX()
taiex = taiex_df.avg               
taiex = taiex.to_numpy()  

'##### MELBOURNE TEMP #####'
melb_temp_df = pd.read_excel('maximum-temp-melbourne.xlsx')
melb_temp = melb_temp_df['Temp']
melb_temp = melb_temp.to_numpy()

'##### ENROLLMENTS #####'
enroll = pd.read_excel('Enrollments.xlsx')
ano = enroll.Year
enroll = enroll.Enrollments.to_numpy()

'##### MILHO #####'
milho_df = pd.read_excel('wheat_production.xlsx')
milho = milho_df['Actual production (kg/ha)']
milho = milho.to_numpy()


'##### SUNSPOT #####'
sun_df = pd.read_csv('sunspots.csv')
sunspots = sun_df.Sunspots
sunspots = sunspots.to_numpy()
#plt.plot(sunspots)
#plt.ylabel("Number of sunspots")
#plt.xlabel("Instance")


'##### SHANGAI AQI #####'
shangai_df = pd.read_excel('shangai_AQI.xlsx')
shan_aqi = shangai_df['AQI']
shan_aqi = shan_aqi.to_numpy()

'##### HANGZHOU AQI #####'
hangzhou_df = pd.read_excel('hangzhou_AQI.xlsx')
han_aqi = hangzhou_df['AQI']
han_aqi = han_aqi.to_numpy()


'##### Klang #####'
'Data completo'
klang_df_geral = pd.read_csv('apims-2005-2017.csv')
klang_api_date_time = klang_df_geral['Time']
klang_api_value = klang_df_geral['Klang']
klang_df = pd.concat([klang_api_date_time, klang_api_value], axis=1)
klang_df_fatiado = klang_df.loc[5361:20268]

'calcular a media diaria'
klang_api_date = klang_df_fatiado['Time'].tolist()
klang_api_valor = klang_df_fatiado['Klang'].tolist()

'tirar a hora da string com a data'
klang_api_date_only = []
for x in klang_api_date:
    klang_api_date_only.append(x[:10])
    
klang_date = pd.Series(klang_api_date_only)
klang_valor = pd.Series(klang_api_valor)   

klang_df_final = pd.concat([klang_date.rename('Time'), klang_valor.rename('Klang')], axis=1)

'agrupa e calcula media diaria'
api_diario = klang_df_final.groupby('Time')[['Klang']].mean()

writer = pd.ExcelWriter('api_klang_diario.xlsx', engine='xlsxwriter')
api_diario.to_excel(writer, sheet_name='Klang')
writer.save()

'###problema: esta com nan'

klang_df = pd.read_excel('klang_api.xlsx')
klang = klang_df['Klang']
klang = klang.to_numpy()

klang[:731]


klang_df = pd.read_csv('kalang.csv')
klang = klang_df['value']
klang = klang.to_numpy()
"""

df_brent_oil = get_Brent_Oil()
brent_oil = df_brent_oil.Price  
brent_oil = brent_oil.to_numpy()

"""
plt.figure(figsize = (8,4))
plt.plot(brent_oil)
plt.xlabel("Samples")
plt.ylabel("Brent Oil Prices (US$/Barrel)")
"""


datasets = [brent_oil]
dataset_names = ['brent_oil']
diff = 1                                #If diff = 1, data is differentiated
partition_parameters = np.arange(5,6)            #partiions must be a list
orders = [1]
partitioners = ['ADP_ANTIGO']                 #partitioners: 'chen' 'SODA' 'ADP' 'DBSCAN' 'CMEANS' 'entropy' 'FCM'  
mfs = ['trapezoidal']                    #mfs: 'triangular' ou 'trapezoidal' ou 'gaussian'


'------------------------------------------------ Running the model -------------------------------------------------'


'Builds and runs the model'
run_Gridsearch(datasets,dataset_names,diff,partition_parameters,orders,partitioners,mfs,training = 0.7)
