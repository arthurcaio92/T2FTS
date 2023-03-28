# -*- coding: utf-8 -*-



import numpy, scipy.io, scipy.spatial.distance, numpy.matlib, numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

#distancetype='chebyshev' #changeable, i.e. 'euclidean','cityblock','sqeuclidean','cosine'.
#granularity=11 #changeable, any positive integer. The larger granularity is, the more details the partitioning result gives.

def plotar_adp(data, centre, IDX):
    
    plt.plot(data['avg'])
    plt.figure()
    adp_idx = IDX
    centros = centre
    df_serie_temporal = data
    df_serie_temporal['Clouds'] = adp_idx
    #print(df_serie_temporal)
    import matplotlib.colors as mcolors    
    cores_basicas = mcolors.BASE_COLORS
    colors1 = list(cores_basicas.values())
    cores_tableau = mcolors.CSS4_COLORS
    colors2 = list(cores_tableau.values())
    colors = colors1[:-2] + colors2[10:60]
    plt.scatter((df_serie_temporal['n']), (df_serie_temporal['avg']), c=(df_serie_temporal['Clouds']), cmap=(mcolors.ListedColormap(colors)), s=4)
    for m in range(len(centros)):
        plt.plot((centros[m][0]), (centros[m][1]), color='black', markersize=8, marker='d')

    plt.figure()
    plt.scatter((df_serie_temporal['avg']), (df_serie_temporal['n']), c=(df_serie_temporal['Clouds']), cmap=(mcolors.ListedColormap(colors)), s=4)
    for m in range(len(centros)):
        plt.plot((centros[m][1]), (centros[m][0]), color='black', markersize=8, marker='d')

    T = np.unique(adp_idx)
    cloud_info = pd.DataFrame()
    lista_max = []
    lista_min = []
    lista_std = []
    lista_qtd = []
    lista_centros = []
    for x in T:
        fatiado = df_serie_temporal.loc[(df_serie_temporal['Clouds'] == x)]
        quantidade = len(fatiado.index)
        if quantidade > 1:
            lista_qtd.append(quantidade)
            maximo = fatiado['avg'].max()
            lista_max.append(maximo)
            minimo = fatiado['avg'].min()
            lista_min.append(minimo)
            desvio = fatiado['avg'].std()
            lista_std.append(desvio)
            lista_centros.append(centros[(x - 1)])

    cloud_info['Cloud'] = np.arange(1, len(lista_qtd) + 1)
    cloud_info['Min'] = lista_min
    cloud_info['Max'] = lista_max
    cloud_info['qtd'] = lista_qtd
    cloud_info['std'] = lista_std
    cloud_info['Centros'] = lista_centros
    elements_count = {}
    for element in adp_idx:
        if element in elements_count:
            elements_count[element] += 1
        else:
            elements_count[element] = 1

    elements_count_ordered = dict(sorted((elements_count.items()), key=(lambda x: (x[1], x[0]))))
    maiores_clouds = list(elements_count_ordered.keys())[-10:]
    return (
     cloud_info, lista_centros)


def ADP(data, granularity=11, distancetype='chebyshev'):
    dados_original = data
    
    data=numpy.float32(numpy.matrix(data))
    L0,W0=data.shape
    udata,frequency = numpy.unique(data,return_counts=True, axis=0)
    frequency=numpy.matrix(frequency)
    L,W=udata.shape
    dist=(scipy.spatial.distance.cdist(udata,udata,metric=distancetype))**2
    unidata_pi=numpy.sum(numpy.multiply(dist,numpy.matlib.repmat(frequency,L,1)), axis=1)
    unidata_density=numpy.transpose(unidata_pi)*numpy.transpose(frequency)/(unidata_pi*2*L0)
    unidata_Gdensity=numpy.multiply(unidata_density,numpy.transpose(frequency))

    pos=numpy.zeros(L0).astype(int)  # se der erro, tira esse astype
    pos[0]=int(numpy.argmax(unidata_Gdensity))
    seq=numpy.array(range(0,L0))
    seq=numpy.delete(seq,pos[0])

    for ii in range(1,L):
        p1=numpy.argmin(dist[int(pos[ii-1]),seq])
        pos[ii]=seq[p1]
        seq=numpy.delete(seq,p1)

    udata2=numpy.zeros([L,W])
    uGD=numpy.zeros(L)
    for ii in range(0,L):
        udata2[ii,:]=udata[int(pos[ii]),:]
        uGD[ii]=unidata_Gdensity[int(pos[ii]),0]

    uGD1=uGD[range(0,L-2)]-uGD[range(1,L-1)]
    uGD2=uGD[range(1,L-1)]-uGD[range(2,L)]
    seq2=numpy.array(range(1,L-1))

    seq3=numpy.array([i for i in range(len(uGD2)) if uGD1[i]<0 and uGD2[i]>0])

    seq4=numpy.array([0])
    if uGD2[L-3]<0:
        seq4=numpy.append(seq4,seq2[seq3])
        seq4=numpy.append(seq4,numpy.array([int(L-1)]))
    else:
        seq4=numpy.append(seq4,seq2[seq3])

    L2, =seq4.shape
    centre0=numpy.zeros([L2,W])
    for ii in range(0,L2):
        centre0[ii]=udata2[int(seq4[ii]),:]
    
    dist1=scipy.spatial.distance.cdist(data,centre0,metric=distancetype)
    seq5=dist1.argmin(1)
    centre1=numpy.zeros([L2,W])
    Mnum=numpy.zeros(L2)

    for ii in range(0,L2):
        seq6=[i for i in range(len(seq5)) if seq5[i] == ii]
        Mnum[ii]=len(seq6)
        centre1[ii,:]=numpy.mean(data[seq6,:],axis=0)

    seq7=[i for i in range(len(Mnum)) if Mnum[i] > 1]
    seq8=[i for i in range(len(Mnum)) if Mnum[i] <= 1]

    L3=len(seq7)
    L4=len(seq8)

    centre2=numpy.zeros([L3,W])
    centre3=numpy.zeros([L4,W])
    Mnum1=numpy.zeros(L3)
    for ii in range(0,L3):
        centre2[ii,:]=centre1[seq7[ii],:]
        Mnum1[ii]=Mnum[seq7[ii]]

    for ii in range(0,L4):
        centre3[ii,:]=centre1[seq8[ii],:]

    dist2=scipy.spatial.distance.cdist(centre3,centre2,distancetype)
    seq9=dist2.argmin(1)

    for ii in range(0,L4):
        centre2[seq9[ii],:]=centre2[seq9[ii],:]*Mnum1[seq9[ii]]/(Mnum1[seq9[ii]]+1)+centre3[ii,:]/(Mnum1[seq9[ii]]+1)
        Mnum1[seq9[ii]]=Mnum1[seq9[ii]]+1

    UD2=centre2
    L5=0
    Count=0
    while L5 != L3 and L3>2:
        Count=Count+1
        L5=L3
        dist3=scipy.spatial.distance.cdist(data,UD2,distancetype)
        seq10=dist3.argmin(1)
        centre3=numpy.zeros([L3,W])
        Mnum3=numpy.zeros(L3)
        Sigma3=numpy.zeros(L3)
        seq12=[]
        for ii in range(0,L3):
            seq11=[i for i in range(len(seq10)) if seq10[i] == ii]
            if len(seq11)>=2:
                data1=data[seq11,:]
                Mnum3[ii]=len(seq11)
                centre3[ii,:]=numpy.sum(data1,axis=0)/Mnum3[ii]
                Sigma3[ii]=numpy.sum(numpy.sum(numpy.multiply(data1,data1)))/Mnum3[ii]-numpy.sum(numpy.multiply(centre3[ii,:],centre3[ii,:]))
                if Sigma3[ii]>0:
                    seq12.append(ii)
        L3=len(seq12)
        Mnum3=numpy.matrix(Mnum3[seq12])
        centre3=centre3[seq12,:]
        dist=(scipy.spatial.distance.cdist(centre3,centre3,distancetype))**2
        unidata_pi=numpy.sum(numpy.multiply(dist,numpy.matlib.repmat(Mnum3,L3,1)), axis=1)
        unidata_density=numpy.transpose(unidata_pi)*numpy.transpose(Mnum3)/(unidata_pi*2*L0)
        unidata_Gdensity=numpy.multiply(unidata_density,numpy.transpose(Mnum3))
        dist2=(scipy.spatial.distance.pdist(centre3,distancetype))
        dist3=scipy.spatial.distance.squareform(dist2)
        Aver1=numpy.mean(dist2)
        for ii in range(granularity):
            Aver1=numpy.mean(dist2[dist2<=Aver1])
        Sigma=Aver1/2
        dist3=dist3-numpy.ones([L3,L3])*Sigma
        seq15=[]
        for i in range(0,L3):
            seq13=numpy.array(list(range(0,i))+list(range(i+1,L3)))
            seq14=seq13[dist3[i,seq13]<0]
            if len(seq14)>0:
                if unidata_Gdensity[i]>max(unidata_Gdensity[seq14]):
                    seq15.append(i)
            else:
                seq15.append(i)
        L3=len(seq15)
        UD2=centre3[numpy.array(seq15),:]

    centre=UD2

    dist1=scipy.spatial.distance.cdist(data,centre,distancetype)
    IDX=dist1.argmin(1)
    Mnum=numpy.zeros(L3)

    for ii in range(0,L3):
        seq6=[i for i in range(len(IDX)) if IDX[i] == ii]
        Mnum[ii]=len(seq6)
        centre[ii,:]=numpy.sum(data[seq6,:],axis=0)/Mnum[ii]
        
    adp_idx = []
    for x in IDX:
        adp_idx.append(x + 1)

    cloud_info, centros = plotar_adp(dados_original, centre, adp_idx)
    
    return (centros, cloud_info)

    
    '------------------'
    '------------------'
    '------------------'
    '------------------'  
    '------------------'
    
def ADP_antigo(data, granularity=11, distancetype='chebyshev'):
    data=numpy.float32(numpy.matrix(data))
    L0,W0=data.shape
    udata,frequency = numpy.unique(data,return_counts=True, axis=0)
    frequency=numpy.matrix(frequency)
    L,W=udata.shape
    dist=(scipy.spatial.distance.cdist(udata,udata,metric=distancetype))**2
    unidata_pi=numpy.sum(numpy.multiply(dist,numpy.matlib.repmat(frequency,L,1)), axis=1)
    unidata_density=numpy.transpose(unidata_pi)*numpy.transpose(frequency)/(unidata_pi*2*L0)
    unidata_Gdensity=numpy.multiply(unidata_density,numpy.transpose(frequency))

    pos=numpy.zeros(L0).astype(int)  # se der erro, tira esse astype
    pos[0]=int(numpy.argmax(unidata_Gdensity))
    seq=numpy.array(range(0,L0))
    seq=numpy.delete(seq,pos[0])

    for ii in range(1,L):
        p1=numpy.argmin(dist[int(pos[ii-1]),seq])
        pos[ii]=seq[p1]
        seq=numpy.delete(seq,p1)

    udata2=numpy.zeros([L,W])
    uGD=numpy.zeros(L)
    for ii in range(0,L):
        udata2[ii,:]=udata[int(pos[ii]),:]
        uGD[ii]=unidata_Gdensity[int(pos[ii]),0]

    uGD1=uGD[range(0,L-2)]-uGD[range(1,L-1)]
    uGD2=uGD[range(1,L-1)]-uGD[range(2,L)]
    seq2=numpy.array(range(1,L-1))

    seq3=numpy.array([i for i in range(len(uGD2)) if uGD1[i]<0 and uGD2[i]>0])

    seq4=numpy.array([0])
    if uGD2[L-3]<0:
        seq4=numpy.append(seq4,seq2[seq3])
        seq4=numpy.append(seq4,numpy.array([int(L-1)]))
    else:
        seq4=numpy.append(seq4,seq2[seq3])

    L2, =seq4.shape
    centre0=numpy.zeros([L2,W])
    for ii in range(0,L2):
        centre0[ii]=udata2[int(seq4[ii]),:]
    
    dist1=scipy.spatial.distance.cdist(data,centre0,metric=distancetype)
    seq5=dist1.argmin(1)
    centre1=numpy.zeros([L2,W])
    Mnum=numpy.zeros(L2)

    for ii in range(0,L2):
        seq6=[i for i in range(len(seq5)) if seq5[i] == ii]
        Mnum[ii]=len(seq6)
        centre1[ii,:]=numpy.mean(data[seq6,:],axis=0)

    seq7=[i for i in range(len(Mnum)) if Mnum[i] > 1]
    seq8=[i for i in range(len(Mnum)) if Mnum[i] <= 1]

    L3=len(seq7)
    L4=len(seq8)

    centre2=numpy.zeros([L3,W])
    centre3=numpy.zeros([L4,W])
    Mnum1=numpy.zeros(L3)
    for ii in range(0,L3):
        centre2[ii,:]=centre1[seq7[ii],:]
        Mnum1[ii]=Mnum[seq7[ii]]

    for ii in range(0,L4):
        centre3[ii,:]=centre1[seq8[ii],:]

    dist2=scipy.spatial.distance.cdist(centre3,centre2,distancetype)
    seq9=dist2.argmin(1)

    for ii in range(0,L4):
        centre2[seq9[ii],:]=centre2[seq9[ii],:]*Mnum1[seq9[ii]]/(Mnum1[seq9[ii]]+1)+centre3[ii,:]/(Mnum1[seq9[ii]]+1)
        Mnum1[seq9[ii]]=Mnum1[seq9[ii]]+1

    UD2=centre2
    L5=0
    Count=0
    while L5 != L3 and L3>2:
        Count=Count+1
        L5=L3
        dist3=scipy.spatial.distance.cdist(data,UD2,distancetype)
        seq10=dist3.argmin(1)
        centre3=numpy.zeros([L3,W])
        Mnum3=numpy.zeros(L3)
        Sigma3=numpy.zeros(L3)
        seq12=[]
        for ii in range(0,L3):
            seq11=[i for i in range(len(seq10)) if seq10[i] == ii]
            if len(seq11)>=2:
                data1=data[seq11,:]
                Mnum3[ii]=len(seq11)
                centre3[ii,:]=numpy.sum(data1,axis=0)/Mnum3[ii]
                Sigma3[ii]=numpy.sum(numpy.sum(numpy.multiply(data1,data1)))/Mnum3[ii]-numpy.sum(numpy.multiply(centre3[ii,:],centre3[ii,:]))
                if Sigma3[ii]>0:
                    seq12.append(ii)
        L3=len(seq12)
        Mnum3=numpy.matrix(Mnum3[seq12])
        centre3=centre3[seq12,:]
        dist=(scipy.spatial.distance.cdist(centre3,centre3,distancetype))**2
        unidata_pi=numpy.sum(numpy.multiply(dist,numpy.matlib.repmat(Mnum3,L3,1)), axis=1)
        unidata_density=numpy.transpose(unidata_pi)*numpy.transpose(Mnum3)/(unidata_pi*2*L0)
        unidata_Gdensity=numpy.multiply(unidata_density,numpy.transpose(Mnum3))
        dist2=(scipy.spatial.distance.pdist(centre3,distancetype))
        dist3=scipy.spatial.distance.squareform(dist2)
        Aver1=numpy.mean(dist2)
        for ii in range(granularity):
            Aver1=numpy.mean(dist2[dist2<=Aver1])
        Sigma=Aver1/2
        dist3=dist3-numpy.ones([L3,L3])*Sigma
        seq15=[]
        for i in range(0,L3):
            seq13=numpy.array(list(range(0,i))+list(range(i+1,L3)))
            seq14=seq13[dist3[i,seq13]<0]
            if len(seq14)>0:
                if unidata_Gdensity[i]>max(unidata_Gdensity[seq14]):
                    seq15.append(i)
            else:
                seq15.append(i)
        L3=len(seq15)
        UD2=centre3[numpy.array(seq15),:]

    centre=UD2

    dist1=scipy.spatial.distance.cdist(data,centre,distancetype)
    IDX=dist1.argmin(1)
    Mnum=numpy.zeros(L3)

    for ii in range(0,L3):
        seq6=[i for i in range(len(IDX)) if IDX[i] == ii]
        Mnum[ii]=len(seq6)
        centre[ii,:]=numpy.sum(data[seq6,:],axis=0)/Mnum[ii]
    return centre,IDX