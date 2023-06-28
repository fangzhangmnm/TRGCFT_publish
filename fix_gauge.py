from opt_einsum import contract
import torch
#from safe_svd import svd,sqrt # TODO is it necessary???
from torch.linalg import svd
from torch import sqrt
from tqdm.auto import tqdm
import dataclasses
from dataclasses import dataclass
from math import prod
import numpy as np
import itertools as it


def dcontract(derivative,eq,*tensors,**kwargs):
    assert all(tensor is not None for tensor in tensors)
    assert len(list(tensor for tensor in tensors if id(tensor)==id(derivative)))==1, f'{id(derivative)%3533} {[id(t)%3533 for t in tensors]}'
    idx = next(i for i, tensor in enumerate(tensors) if id(tensor)==id(derivative))
    eq_terms=eq.split(',')
    eq=','.join(eq_terms[:idx]+eq_terms[idx+1:])+'->'+eq_terms[idx]
    tensors=tensors[:idx]+tensors[idx+1:]
    return contract(eq,*tensors,**kwargs)

def svd_tensor_to_isometry(M,idx1=(0,),idx2=(1,)):
    shape1=tuple(M.shape[i] for i in idx1)
    shape2=tuple(M.shape[i] for i in idx2)
    M=M.permute(idx1+idx2).reshape(prod(shape1),prod(shape2))
    u,_,vh=torch.linalg.svd(M,full_matrices=False,driver='gesvd')
    uvh=(u@vh).reshape(shape1+shape2).permute(tuple(np.argsort(idx1+idx2)))
    return uvh

def get_isometry_from_environment(M,idx1=(0,),idx2=(1,)):
    return svd_tensor_to_isometry(M,idx1,idx2).conj()

    
def contract_all_legs(T1,T2:torch.Tensor)->torch.Tensor:
    T1i,T2i=[*range(len(T1.shape))],[*range(len(T2.shape))]
    return contract(T1,T1i,T2,T2i)

def contract_all_legs_but_one(T1,T2:torch.Tensor,i:int)->torch.Tensor:
    T1i,T2i=[*range(len(T1.shape))],[*range(len(T2.shape))]
    T1i[i],T2i[i]=-1,-2
    return contract(T1,T1i,T2,T2i,[-1,-2])

def sum_all_legs_but_one(T,i:int)->torch.Tensor:
    Ti=[*range(len(T.shape))]
    return contract(T,Ti,[i])

def apply_matrix_to_leg(T:torch.Tensor,M:torch.Tensor,i:int)->torch.Tensor:
    Ti,Mi=[*range(len(T.shape))],[-1,i]
    Tni=Ti.copy();Tni[i]=-1
    return contract(T,Ti,M,Mi,Tni)

def apply_vector_to_leg(T:torch.Tensor,M:torch.Tensor,i:int)->torch.Tensor:
    Ti,Mi=[*range(len(T.shape))],[i]
    Tni=Ti.copy()
    return contract(T,Ti,M,Mi,Tni)


@dataclass
class MCF_options:
    enabled:bool=True
    eps:float=1e-6
    max_iter:int=50
    fix_unitary_enabled:bool=True
    phase_iter1:int=3
    phase_iter2:int=10


# def fix_unitary_gauge_svd_2D(T,Tref,nIter=10):
#     dim1,dim2=T.shape[0],T.shape[2]
#     h1,h2=torch.eye(dim1),torch.eye(dim2)
#     for i in range(nIter):
#         env_h1=dcontract(h1,'ijkl,IJKL,Ii,Jj,Kk,Ll',T,Tref,h1,h1.conj().clone(),h2,h2.conj().clone())
#         h1=get_isometry_from_environment(env_h1)
#         env_h2=dcontract(h2,'ijkl,IJKL,Ii,Jj,Kk,Ll',T,Tref,h1,h1.conj().clone(),h2,h2.conj().clone())
#         h2=get_isometry_from_environment(env_h2)
#     return contract('ijkl,Ii,Jj,Kk,Ll->IJKL',T,h1,h1.conj(),h2,h2.conj()),[h1,h1.conj(),h2,h2.conj()]


# def fix_unitary_gauge_main_2D(T,Tref,options:MCF_options=MCF_options()):
#     hs=[torch.eye(T.shape[i]) for i in range(4)]
#     if Tref is not None and T.shape==Tref.shape:
#         for _j in range(options.phase_iter1):
#             ds=[torch.ones(T.shape[i]) for i in range(4)]
#             for _i in range(1,max(T.shape)):
#                 i=_i%max(T.shape)
#                 if len(T.shape)==4:
#                     TT,TTref=T[:,:i,:i,:i],Tref[:,:i,:i,:i]
#                 else:
#                     TT,TTref=T[:,:i,:i,:i,0],Tref[:,:i,:i,:i,0]
#                 rho1=contract('ijkl,ijkl->i',TT,TTref)
#                 di=torch.where(rho1>0,1.,-1.)
#                 ds[0],ds[1]=ds[0]*di,ds[1]*di
#                 #T=contract('ijkl,i,j->ijkl',T,di,di)
#                 T=apply_vector_to_leg(T,di,0)
#                 T=apply_vector_to_leg(T,di,1)
                
#                 if len(T.shape)==4:
#                     TT,TTref=T[:i,:i,:,:i],Tref[:i,:i,:,:i]
#                 else:
#                     TT,TTref=T[:i,:i,:,:i,0],Tref[:i,:i,:,:i,0]
#                 rho1=contract('ijkl,ijkl->k',TT,TTref)
#                 di=torch.where(rho1>0,1.,-1.)
#                 ds[2],ds[3]=ds[2]*di,ds[3]*di
#                 #T=contract('ijkl,k,l->ijkl',T,di,di)
#                 T=apply_vector_to_leg(T,di,2)
#                 T=apply_vector_to_leg(T,di,3)

#             hs=[torch.diag(di)@h for di,h in zip(ds,hs)]
#             T,hs1=fix_unitary_gauge_svd(T,Tref,nIter=options.phase_iter2)
#             hs=[h1@h for h1,h in zip(hs1,hs)]
#     return T,hs

# def fix_unitary_gauge(T,Tref,options:MCF_options=MCF_options()):
#     foo={4:fix_unitary_gauge_main_2D}[len(T.shape)]
#     return foo(T,Tref,options=options)

def fix_unitary_gauge_svd(T,Tref,nIter=10):
    spacial_dim=len(T.shape)//2
    hh=[torch.eye(T.shape[i]) for i in range(2*spacial_dim)]
    eq1='ijklmn'[:2*spacial_dim]
    eq2=eq1.upper()
    eq3='Ii,Jj,Kk,Ll,Mm,Nn'[:6*spacial_dim-1]

    for _i in range(nIter):
        for iDim in range(spacial_dim):
            env_h=dcontract(hh[2*iDim],eq1+','+eq2+','+eq3,T,Tref,*hh)
            hh[2*iDim]=get_isometry_from_environment(env_h)
            hh[2*iDim+1]=hh[2*iDim].conj().clone()
    return contract(eq1+','+eq3+'->'+eq2,T,*hh),hh

def fix_unitary_gauge(T,Tref,options:MCF_options=MCF_options()):
    spacial_dim=len(T.shape)//2
    hs=[torch.eye(T.shape[i]) for i in range(spacial_dim*2)]
    if options.fix_unitary_enabled:
        # print('fix_unitary_gauge')
        if Tref is not None and T.shape==Tref.shape:
            for iIter1 in range(options.phase_iter1):
                ds=[torch.ones(T.shape[i]) for i in range(2*spacial_dim)]
                for iIter2 in range(1,max(T.shape)):
                    iBondDim=iIter2%max(T.shape)
                    for iDim in range(spacial_dim):
                        slices=[slice(iBondDim)]*2*spacial_dim;slices[2*iDim]=slice(None)
                        TT,TTref=T[slices],Tref[slices] # :,:i,:i,:i,:i,:i or :i,:i,:,:i,:i,:i or :i,:i,:i,:i,:,:i
                        eq1='ijklmn'[:2*spacial_dim]
                        eq2='ijklmn'[2*iDim]
                        rho=contract(eq1+','+eq1+'->'+eq2,TT,TTref)# ijklmn,ijklmn->i or k or m
                        di=torch.where(rho>0,1.,-1.)
                        ds[2*iDim],ds[2*iDim+1]=ds[2*iDim]*di,ds[2*iDim+1]*di
                        T=apply_vector_to_leg(T,di,2*iDim)
                        T=apply_vector_to_leg(T,di,2*iDim+1)

                hs=[torch.diag(di)@h for di,h in zip(ds,hs)]
                T,hs1=fix_unitary_gauge_svd(T,Tref,nIter=options.phase_iter2)
                hs=[h1@h for h1,h in zip(hs1,hs)]
    return T,hs

def minimal_canonical_form(T:torch.Tensor,options:MCF_options=MCF_options())->'tuple[torch.Tensor,list[torch.Tensor]]':
        # The minimal canonical form of a tensor network
        # https://arxiv.org/pdf/2209.14358.pdf
    spacial_dim=len(T.shape)//2
    hh=[torch.eye(T.shape[i]) for i in range(spacial_dim*2)]
    if options.enabled:
        for iIter in range(options.max_iter):
            total_diff=0
            for k in range(spacial_dim):
                tr_rho=contract_all_legs(T,T.conj())
                rho1=contract_all_legs_but_one(T,T.conj(),2*k)
                rho2=contract_all_legs_but_one(T,T.conj(),2*k+1).T
                rho_diff=rho1-rho2
                assert (rho_diff-rho_diff.T.conj()).norm()/tr_rho<1e-7
                total_diff+=rho_diff.norm()**2/tr_rho
                g1=torch.matrix_exp(-rho_diff/(4*spacial_dim*tr_rho))
                g2=torch.matrix_exp(rho_diff/(4*spacial_dim*tr_rho)).T
                hh[2*k]=g1@hh[2*k]
                hh[2*k+1]=g2@hh[2*k+1]
                T=apply_matrix_to_leg(T,g1,2*k)
                T=apply_matrix_to_leg(T,g2,2*k+1)
            if total_diff<options.eps**2:
                break
    return T,hh
    
    
# it seems unitary is already restored because rho_diff is always zero
# def fix_unitary_gauge_other(T,Tref,options:MCF_options=MCF_options()):
#     spacial_dim=len(T.shape)//2
#     hh=[torch.eye(T.shape[i]) for i in range(spacial_dim*2)]
#     for iIter in range(options.max_iter):
#         total_diff=0
#         for k in range(spacial_dim):
#             tr_rho=contract_all_legs(T,Tref.conj())
#             rho1=contract_all_legs_but_one(T,Tref.conj(),2*k)
#             rho2=contract_all_legs_but_one(T,Tref.conj(),2*k).T
#             rho_diff=rho1-rho2
#             print(rho_diff.norm()/tr_rho)
#             assert (rho_diff+rho_diff.T.conj()).norm()/tr_rho<1e-7
#             rho_diff=(rho_diff-rho_diff.T.conj())/2
#             total_diff+=rho_diff.norm()**2/tr_rho
#             g1=torch.matrix_exp(-rho_diff/(4*spacial_dim*tr_rho))
#             g2=g1
#             #g2=torch.matrix_exp(rho_diff/(4*spacial_dim*tr_rho))
#             hh[2*k]=g1@hh[2*k]
#             hh[2*k+1]=g2@hh[2*k+1]
#             T=apply_matrix_to_leg(T,g1,2*k)
#             T=apply_matrix_to_leg(T,g2,2*k+1)
#         if total_diff<options.eps**2:
#             break
#     return T,hh
    
    


    
