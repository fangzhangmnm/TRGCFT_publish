import torch
import numpy as np
from tqdm.auto import tqdm as tqdm
from opt_einsum import contract

#from safe_svd import svd,sqrt # TODO is it necessary???
from torch.linalg import svd


import importlib
from HOTRG import HOTRGLayer

svd_driver='gesvd' #we need precision and consistency, so use gesvd instead of gesvdj

def get_w_HOSVD(MM:torch.Tensor,max_dim):
    # w MM wh
    #S,U=torch.linalg.eigh(MM)#ascending, U S Uh=MM #will fail when there's a lot of zero eigenvalues
    #S,U=S.flip(0),U.flip(-1)
    U,S,Vh=svd(MM,driver=svd_driver) #descending, U S Vh=MM #we need precision and consistency, so use gesvd instead of gesvdj
    w=(U.T)[:max_dim]
    return w



def _HOSVD_layer_3D(T1,T2,max_dim):
    MM1=contract('ijklmn,jopqrs,itulmn,tovqrs->kpuv',T1,T2,T1.conj(),T2.conj())
    MM2=contract('ijklmn,jopqrs,itklun,topqvs->mruv',T1,T2,T1.conj(),T2.conj())
    MM1=MM1.reshape(T1.shape[2]*T2.shape[2],-1)
    MM2=MM2.reshape(T1.shape[4]*T2.shape[4],-1)

    w1=get_w_HOSVD(MM1,max_dim=max_dim)
    wP1=w1.reshape(-1,T1.shape[2],T2.shape[2])
    w2=get_w_HOSVD(MM2,max_dim=max_dim)
    wP2=w2.reshape(-1,T1.shape[4],T2.shape[4])

    Tn=contract('ijklmn,jopqrs,akp,blq,cmr,dns->abcdio',T1,T2,wP1,wP1.conj(),wP2,wP2.conj())
    return Tn,HOTRGLayer(tensor_shape=T1.shape,ww=[w1,w2])
    
def _HOSVD_layer_2D(T1,T2,max_dim):
    MM=contract('ijkl,jmno,ipql,pmro->knqr',T1,T2,T1.conj(),T2.conj())
    MM=MM.reshape(T1.shape[2]*T2.shape[2],-1)

    w=get_w_HOSVD(MM,max_dim=max_dim)
    wP=w.reshape(-1,T1.shape[2],T2.shape[2])
        
    Tn=contract('ijkl,jmno,akn,blo->abim',T1,T2,wP,wP.conj())
    return Tn,HOTRGLayer(tensor_shape=T1.shape,ww=[w])

def HOSVD_layer(T1,T2,max_dim)->"tuple[torch.Tensor,HOTRGLayer]":
    _HOSVD_layer={4:_HOSVD_layer_2D,6:_HOSVD_layer_3D}[len(T1.shape)]
    return _HOSVD_layer(T1,T2,max_dim=max_dim)



