# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 21:26:16 2023

@author: arthu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

from T2FTS import Type2Model,FuzzySet, tri_mf, trapezoid_mf, ltri_mf,rtri_mf
from T2FTS import Type2Model,IT2FS_plot,IT2FS_plot_OLD,min_t_norm, max_s_norm, TR_plot, crisp,tri_mf

def calcula_dominio(dados,var,numero_de_sets):
  """
  Função que recebe os valores de treino de uma variavel e cria o universo de discurso (dominio) para ela

  :params:
  dados: Dataset completo com todos os dados de todas as variaveis
  var: Variavel a ser analisada por esta função
  numero_de_sets: Num. de intervalos em que o universo será particionado

  :return:
  domain: Dominio da variavel analisada
  intervalos_conjuntos: Intervalos para serem construídos cada um dos conjuntos fuzzy. Esta variável só é usada 
  na função cria_conjuntos. Nas outras ela deve ser IGNORADA
  """

  minimo = min(dados[var])
  maximo = max(dados[var])
   
  dominio_inf = np.float64(minimo * 1.1 if minimo < 0 else minimo * 0.9)
  dominio_sup = np.float64(maximo * 1.1 if maximo > 0 else maximo * 0.9)

  pontos = dominio_sup - dominio_inf
  if pontos > 1000: 
      domain = np.linspace(dominio_inf, dominio_sup, int(pontos))
  else:
      domain = np.linspace(dominio_inf, dominio_sup, 500)

  intervalo_entre_set = pontos/(numero_de_sets+1) #Points where each fuzzy sets begins       

  'Finds the beggining points of each interval'
  pontos_conj = []
  for x in range(numero_de_sets+2):
      pontos_conj.append(dominio_inf + (intervalo_entre_set * x))
  
  'Builds the intervals'
  intervalos_conjuntos = []
  for x in range(1,numero_de_sets+1):
      aux = [pontos_conj[x-1],pontos_conj[x],pontos_conj[x+1]]
      intervalos_conjuntos.append(aux)

  #print('var',var,domain)

  return domain,intervalos_conjuntos

  



def adiciona_variaveis(modelo,dados,numero_de_sets):

  """
  Adiciona antecedentes e consequente ao modelo, além de definir o dominio para cada variavel

  :params:
  dados: Dataset completo com todos os dados de todas as variaveis
  numero_de_sets: Num. de conjuntos em que o universo será particionado

  :return:
  var_antec: dict em que as chaves são as variaveis antecedentes e os valores são o objeto da variável na biblioteca sktfuzzy
  var_cons: dict em que a chave é a variavel consequente e o valor é o objeto da variável na biblioteca sktfuzzy
  var_geral: dict com todas as variaveis (antecedentes e consequente). A chave é a variavel e o valor é o objeto da variável

  """
  var_cons = dict()
  var_antec = dict()

  for col in dados.columns:                 #Cada col é um nome de variavel
      ma = max(dados[col])
      mi = min(dados[col])    
      if (col == 'Preço'):                  #Preço é o consequente
        modelo.add_output_variable(col)     #Adiciona ao dict de consequentes
      else: #antecedentes
        modelo.add_input_variable(col)      #Adiciona ao dict de antecedentes
      
  modelo_com_variaveis = modelo

  return modelo_com_variaveis


def cria_conjuntos(modelo,dados,particoes):

    """
    Cria os conjuntos fuzzy de forma uniforme para cada variável antecedente e consequente

    :return:
    var_conjuntos: dict em que chave é a variavel e valor é outro dict com cada conjunto dessa variável
    """

    numero_de_sets = particoes

    'Adiciona variáveis ao modelo'
    modelo_com_variaveis = adiciona_variaveis(modelo,dados,numero_de_sets)

    # Particionamento do Universo de Discurso
    'Divide o Universo de Discurso das variáveis e cria conjuntos triangulares'
     
    dict_var_cons = {}
    dict_var_antec = {}

    for col in dados.columns:
        
        dominio, intervalos_conjuntos = calcula_dominio(dados,col,numero_de_sets)
            
        'Builds each fuzzy set by advancing in the list: intervalos_conjuntos '

        if (col == 'Preço'): #consequente
            dict_sets_cons = {}
            for x in range(1,numero_de_sets+1):
      
                r,t,y = intervalos_conjuntos[x-1]
                b_esq = r        #Left endpoint
                topo_tri = t     #Top of triangle
                b_dir = y  

                fou_right = (b_dir-topo_tri)*0.4        #FOU cannot be greater than MF endpoints
                fou_left = (topo_tri-b_esq)*0.4       
                #fou = min(fou_left,fou_right)

                nome = 'A{} {}'.format(x,col)   #Name of the set
                dict_sets_cons[nome] = FuzzySet(dominio, tri_mf, [b_esq, topo_tri, b_dir, 1],tri_mf, [b_esq+fou_left, topo_tri, b_dir-fou_right, 0.9],nome = nome)

            dict_var_cons[col] = dict_sets_cons
   
        else:   #antecedentes
            dict_sets_antec = {}
            for x in range(1,numero_de_sets+1):
      
                r,t,y = intervalos_conjuntos[x-1]
                b_esq = r        #Left endpoint
                topo_tri = t     #Top of triangle
                b_dir = y  

                fou_right = (b_dir-topo_tri)*0.4        #FOU cannot be greater than MF endpoints
                fou_left = (topo_tri-b_esq)*0.4       
                #fou = min(fou_left,fou_right)

                nome = 'A{} {}'.format(x,col)   #Name of the set
                dict_sets_antec[nome] = FuzzySet(dominio, tri_mf, [b_esq, topo_tri, b_dir, 1],tri_mf, [b_esq+fou_left, topo_tri, b_dir-fou_right, 0.9],nome = nome)

            dict_var_antec[col] = dict_sets_antec


    var_geral = {**dict_var_antec,**dict_var_cons}
          
    return dict_var_antec,dict_sets_cons,var_geral,modelo_com_variaveis



def extrai_regras(modelo,dados,dict_var_geral):
    """
    Função para treinar o modelo com os dados do dataset original. As regras a partir da fuzzificação de todos os dados.

    :return:
    lista_regras: Lista com todas as regras

    """

    regras = []
    pertinencias = {}
    conj_pert = []

    for i in range(0, 5):                                    #Itera sobre cada linha do dataset
      pertinencias = {}                                               #dict-> chaves: variavel, valores: tuple com (conjunto,pertinencia)
      for variavel in dict_var_geral.keys():                          #Itera sobre todas as variaveis na linha do dataset
          valor = dados.iloc[i][variavel]                             #Pega o valor da variavel do momento
          max_pert_u = 0.0
          max_pert_l = 0.0
          for nome_conjunto,conjunto in dict_var_geral[variavel].items():            #Itera sobre todos os termos(conjuntos) da variavel analisada
            u = conjunto.umf(valor, conjunto.umf_params) 
            l = conjunto.lmf(valor, conjunto.lmf_params)     #calculates membership of the x value IN LMF     
            #print('naooo entreiii',valor,variavel,conjunto,u,l)
            if u > max_pert_u:                                  #Caso o valor ative mais de um conjunto, pega o conjunto de maior pertinencia   
              if l > max_pert_l:     
                #print('entreiii',valor,variavel,conjunto,u,l)                                   
                max_pert_u = u
                max_pert_l = l
                pertinencias[variavel] = (nome_conjunto,conjunto,max_pert_u,max_pert_l) 
      regras.append(pertinencias)                         #lista contendo dicts. Cada dict é uma regra composto por -> chaves: variavel, valores: tuple com (conjunto,pertinencia)

    modelo_var_reg = adiciona_regras(modelo,regras)                 

    return modelo_var_reg

def adiciona_regras(modelo,regras):
    """
    Adiciona automaticamente as regras ao modelo

    :params:
    regras: lista contendo dicts. Cada dict é uma regra composto por -> chaves: variavel, valores: tuple com (conjunto,pertinencia)
    :return:
    lista_regras: lista em que cada elemento é uma regra no formato da biblioteca scikit fuzzy

    """
    lista_prop_antec = []
    lista_prop_cons = []
    lista_regras = []                            
    for regra in regras:                                #Itera sobre cada regra
      for variavel,conjunto in regra.items():           #Itera sobre cada item do dict atual
        if variavel == 'Preço':     
          proposicao_cons = (conjunto[0],conjunto[1])    #Adiciona consequente à regra. conjunto[0] = nome_conj e conjunto[1] = objeto conjunto              
          lista_prop_cons.append(proposicao_cons)
        else:
          proposicao_antec = [(conjunto[0],conjunto[1])]    #Adiciona antecedente à regra. conjunto[0] = nome_conj e conjunto[1] = objeto conjunto              
          lista_prop_antec.append(proposicao_antec)
      modelo.add_rule(lista_prop_antec,lista_prop_cons)
      
      #lista_regras.append(regra_x)                      #Constroi lista em cada elemento é uma regra. Sera usado na criação da base de regras
        
    return modelo

def confere_resultados(modelo,rent_df,var_geral):

    lista_sim_vals = []                                             #Lista com valores crisp a serem usados na simulação

    'Vamos construir a lista de valores crisp dos antecedentes'
    for i in range(0, 5):                                #Itera sobre cada linha do dataset
        sim_vals = {}
        for variavel in var_geral.keys():                           #Itera sobre todos os antecedentes na linha do dataset
            if variavel != 'Preço':                                 #Preço é consequente então não entra na analise
              valor_crisp = rent_df.iloc[i][variavel]        #Pega o valor crisp da variavel
              sim_vals[variavel] = np.round(valor_crisp,decimals=3)                #Arredonda os valores para 3 casas decimais (importante para longitude e longitude)
        lista_sim_vals.append(sim_vals)                             #Lista com dicts no formato-> variavel:valor

    lista_resultados = []
    for inputs in lista_sim_vals:                               #Cada inputs é uma linha diferente do dataset, com diferentes valores para cada variavel, portanto uma simulação diferente
        it2out, tr = modelo.evaluate_FLS(inputs, min_t_norm, max_s_norm, list(var_geral['Preço'].values())[0].domain, algorithm="EIASC")     #Realiza a inferência
        resultado = crisp(tr["y1"])                           #'Encontra o valor defuzificado' 
        lista_resultados.append(resultado)                    #Guarda em uma lista de resultados            

    preco_previsto = np.array(lista_resultados).round(decimals=0)            #Lista com todos os preços previstos
    preco_real = rent_df['Preço'].to_numpy()                                 #Lista com todos os preços reais

    #Subtração normal entre os vetores
    #erros = np.subtract(preco_real,preco_previsto.round(decimals=0))   #Arredonda os erros previstos e calcula a diferença

    #rmse1 = sqrt(mean_squared_error(preco_real, preco_previsto))
    #rmse2 = mean_squared_error(preco_real, preco_previsto, squared=False)
    rmse = np.sqrt(mean_squared_error(preco_real, preco_previsto))
    
    return rmse

'-------#####################################################-------------'

rent_df = pd.read_excel("data/rent_apart.xlsx")

# Retirando as colunas que não serão usadas

rent_df = rent_df.drop(['Tipo','level_0','index'], axis= 1)
"""                       
rent_df = rent_df.drop(['Tipo','level_0','index','Latitude','Longitude'], axis= 1)

rent_df = rent_df.drop(['Banheiros','Condomínio','IPTU'], axis= 1)
"""

'-------#####################################################-------------'


'Voce deve definir o numero de conjuntos a serem criados'

conjuntos = 5

modelo = Type2Model([1,2,3],1) 



#Adiciona as variaveis antecedentes e consequente e cria os conjuntos fuzzy para cada uma '
dict_var_antec,dict_var_cons,var_geral,modelo_com_variaveis = cria_conjuntos(modelo,rent_df,conjuntos)
#Extrai as regras do dataset e gera uma lista de regras'
modelo_var_reg = extrai_regras(modelo_com_variaveis,rent_df,var_geral)
#Roda cada linha dataset como se fosse uma simulação para obter o RMSE de previsao'
erro_rmse = confere_resultados(modelo_var_reg,rent_df,var_geral)

