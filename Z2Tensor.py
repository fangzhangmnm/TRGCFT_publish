import torch
import itertools as itt
from opt_einsum import contract
from torch.linalg import svd

#======================== Z2 =================================

def RepDim(dimV1R1,dimV1R2,dimV2R1,dimV2R2):
    return (dimV1R1*dimV2R1+dimV1R2*dimV2R2,dimV1R1*dimV2R2+dimV1R2*dimV2R1)

def RepMat(dimV1R1,dimV1R2,dimV2R1=None,dimV2R2=None):
    dimV2R1,dimV2R2=(dimV2R1 or dimV1R1),(dimV2R2 or dimV1R2)
    dimV1=dimV1R1+dimV1R2
    dimV2=dimV2R1+dimV2R2
    P=torch.zeros([dimV1*dimV2,dimV1,dimV2])
    counter=0
    for i in range(dimV1R1):
        for j in range(dimV2R1):
            P[counter,i,j]=1
            counter+=1
    for i in range(dimV1R2):
        for j in range(dimV2R2):
            P[counter,dimV1R1+i,dimV2R1+j]=1
            counter+=1
    for i in range(dimV1R1):
        for j in range(dimV2R2):
            P[counter,i,dimV2R1+j]=1
            counter+=1
    for i in range(dimV1R2):
        for j in range(dimV2R1):
            P[counter,dimV1R1+i,j]=1
            counter+=1
    return P # aij [dimV1*dimV2,dimV1,dimV2]


def Z2_sectors(T,dimR:'tuple[tuple[int]]'):
    dimR=_check_dimR(T,dimR)
    for sector in itt.product(range(2),repeat=len(dimR)):
        begin=[sum(dimR[leg][:rep]) for leg,rep in enumerate(sector)]
        end=[sum(dimR[leg][:rep+1]) for leg,rep in enumerate(sector)]
        slices=[slice(b,e) for b,e in zip(begin,end)]
        yield sector,slices

def Z2_sector_norm(T,dimR:'tuple[tuple[int]]'):
    dimR=_check_dimR(T,dimR)
    sqrnorm=torch.zeros(2)
    for sector,slices in Z2_sectors(T,dimR):
        sqrnorm[sum(sector)%2]+=T[slices].norm()**2
    return sqrnorm**.5

def project_Z2(T,dimR:'tuple[tuple[int]]',weights=[1,0],tolerance=float('inf')):
    dimR=_check_dimR(T,dimR)
    Tn=torch.zeros(T.shape)
    sqrnorm=torch.zeros(2)
    for sector,slices in Z2_sectors(T,dimR):
        sqrnorm[sum(sector)%2]+=T[slices].norm()**2
        Tn[slices]=T[slices]*weights[sum(sector)%2]
    norm=sqrnorm**.5
    assert not(weights[1]==0 and norm[1]>norm[0]*tolerance)
    assert not(weights[0]==0 and norm[0]>norm[1]*tolerance)
    return Tn

def merge_legs_Z2(T,i,j,dimR:'tuple[tuple[int]]'):
    dimR=_check_dimR(T,dimR)
    assert i!=j
    if i>j: i,j=j,i
    P=RepMat(dimR[i][0],dimR[i][1],dimR[j][0],dimR[j][1])
    T_idx=tuple(range(len(T.shape)))
    T_idx_new=T_idx[:i]+(-1,)+T_idx[i+1:j]+T_idx[j+1:]
    T=contract(T,T_idx,P,(-1,i,j,),T_idx_new)
    dimRnew=dimR[:i]+ (RepDim(dimR[i][0],dimR[i][1],dimR[j][0],dimR[j][1]),) +dimR[i+1:j]+dimR[j+1:]
    return T,dimRnew

def svd_Z2(M,dimR_leg:'tuple[int]',**args):
    assert len(M.shape)==2
    dimR=_check_dimR(M,(dimR_leg,))
    assert Z2_sector_norm(M,dimR)[1]==0, 'M is not Z2 symmetric'
    start,mid,end=0,dimR_leg[0],dimR_leg[0]+dimR_leg[1]
    u1,s1,vh1=svd(M[:mid,:mid],**args)
    u2,s2,vh2=svd(M[mid:,mid:],**args)
    u=torch.block_diag(u1,u2)
    s=torch.cat((s1,s2))
    vh=torch.block_diag(vh1,vh2)
    return u,s,vh

def truncate_svd_result_Z2(u,s,vh,dimR_leg:'tuple[int]',max_dim):
    start,mid,end=0,dimR_leg[0],dimR_leg[0]+dimR_leg[1]
    idx=torch.argsort(s,descending=True)
    idx=idx[:max_dim]
    idx1,idx2=idx[idx<mid],idx[idx>=mid]
    dimR_leg_new=(len(idx1),len(idx2))
    idx=torch.cat((idx1,idx2))
    u_new,s_new,vh_new=u[:,idx],s[idx],vh[idx,:]
    return u_new,s_new,vh_new,dimR_leg_new











def _check_dimR(T,dimR:'tuple[tuple[int]]'):
    if len(T.shape)==2*len(dimR): dimR=tuple(d for d in dimR for _ in range(2))
    assert len(T.shape)==len(dimR) and all(i==sum(j) for i,j in zip(T.shape,dimR))
    return dimR