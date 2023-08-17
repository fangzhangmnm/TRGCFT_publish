import torch
import numpy as np
from tqdm.auto import tqdm as tqdm
from opt_einsum import contract

#from safe_svd import svd,sqrt # TODO is it necessary???
from torch.linalg import svd


import importlib
from HOTRG import HOTRGLayer
from Z2Tensor import svd_Z2,merge_legs_Z2,check_dimR

svd_driver='gesvd' #we need precision and consistency, so use gesvd instead of gesvdj

def get_w_HOSVD(MM:torch.Tensor,dimR:'tuple[tuple[int]]',max_dim):
    # w MM wh
    # don't use eigh because it will fail when there's a lot of zero eigenvalues
    # svd gives descending, U S Vh=MM 
    # we need precision and consistency, so use gesvd instead of gesvdj
    "returns w,dimR_w with indices 'aij'"
    check_dimR(MM,dimR)

    U,S,Vh,dimR_u=svd_Z2(MM,dimR_leg=dimR[0],driver=svd_driver,max_dim=max_dim)
    w,dimR_w=U.T,(dimR_u[1],dimR_u[0])
    
    return w,dimR_w

def _HOSVD_layer_2D(T1,T2,dimR:'tuple[tuple[int]]',max_dim):
    assert check_dimR(T1,dimR) and check_dimR(T2,dimR)
    MM=contract('ijkl,jmno,ipql,pmro->knqr',T1,T2,T1.conj(),T2.conj())
    dimR_MM=(dimR[2],dimR[2],dimR[2],dimR[2]) # 'knqr'
    MM,dimR_MM,P,dimR_P=merge_legs_Z2(MM,dimR_MM,0,1)
    MM,dimR_MM,_,_=merge_legs_Z2(MM,dimR_MM,1,2)

    w,dimR_w=get_w_HOSVD(MM,dimR_MM,max_dim=max_dim) # "aij"
        
    Tn=contract('ijkl,jmno,akn,blo->abim',T1,T2,w,w.conj())
    dimR_next=(dimR_w[0],dimR_w[0],dimR[0],dimR[0]) # "abim"
    return Tn,HOTRGLayer(tensor_shape=T1.shape,ww=[w],dimR=dimR,dimR_next=dimR_next)


def _HOSVD_layer_3D(T1,T2,dimR:'tuple[tuple[int]]',max_dim):
    assert check_dimR(T1,dimR) and check_dimR(T2,dimR)
    MM1=contract('ijklmn,jopqrs,itulmn,tovqrs->kpuv',T1,T2,T1.conj(),T2.conj())
    MM2=contract('ijklmn,jopqrs,itklun,topqvs->mruv',T1,T2,T1.conj(),T2.conj())
    dimR_MM1=(dimR[2],dimR[2],dimR[2],dimR[2]) # 'kpuv'
    dimR_MM2=(dimR[4],dimR[4],dimR[4],dimR[4]) # 'mruv'
    MM1,dimR_MM1,P1,dimR_P1=merge_legs_Z2(MM1,dimR_MM1,0,1)
    MM1,dimR_MM1,_,_=merge_legs_Z2(MM1,dimR_MM1,1,2)
    MM2,dimR_MM2,P2,dimR_P2=merge_legs_Z2(MM2,dimR_MM2,0,1)
    MM2,dimR_MM2,_,_=merge_legs_Z2(MM2,dimR_MM2,1,2)

    w1,dimR_w1=get_w_HOSVD(MM1,dimR_MM1,max_dim=max_dim) # "aij"
    w2,dimR_w2=get_w_HOSVD(MM2,dimR_MM2,max_dim=max_dim)

    Tn=contract('ijklmn,jopqrs,akp,blq,cmr,dns->abcdio',T1,T2,w1,w1.conj(),w2,w2.conj())
    dimR_next=(dimR_w1[0],dimR_w1[0],dimR_w2[0],dimR_w2[0],dimR[0],dimR[0]) # "abcdio"
    return Tn,HOTRGLayer(tensor_shape=T1.shape,ww=[w1,w2],dimR=dimR,dimR_next=dimR_next)
    

def HOSVD_layer(T1,T2,dimR:'tuple[tuple[int]]',max_dim):
    "returns Tn,HOTRGLayer"
    _HOSVD_layer={4:_HOSVD_layer_2D,6:_HOSVD_layer_3D}[len(T1.shape)]
    return _HOSVD_layer(T1,T2,dimR=dimR,max_dim=max_dim)



