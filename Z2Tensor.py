import torch
import itertools as itt
from opt_einsum import contract
from torch.linalg import svd

#======================== Z2 =================================




def Z2_sectors(T,dimR:'tuple[tuple[int]]'):
    assert check_dimR(T,dimR)
    for sector in itt.product(range(2),repeat=len(dimR)):
        begin=[sum(dimR[leg][:rep]) for leg,rep in enumerate(sector)]
        end=[sum(dimR[leg][:rep+1]) for leg,rep in enumerate(sector)]
        slices=[slice(b,e) for b,e in zip(begin,end)]
        yield sector,slices

def Z2_sector_norm(T,dimR:'tuple[tuple[int]]'):
    assert check_dimR(T,dimR)
    sqrnorm=torch.zeros(2)
    for sector,slices in Z2_sectors(T,dimR):
        sqrnorm[sum(sector)%2]+=T[slices].norm()**2
    return sqrnorm**.5

def project_Z2(T,dimR:'tuple[tuple[int]]',weights=[1,0],tolerance=float('inf')):
    assert check_dimR(T,dimR)
    Tn=torch.zeros(T.shape)
    sqrnorm=torch.zeros(2)
    for sector,slices in Z2_sectors(T,dimR):
        sqrnorm[sum(sector)%2]+=T[slices].norm()**2
        Tn[slices]=T[slices]*weights[sum(sector)%2]
    norm=sqrnorm**.5
    assert not(weights[1]==0 and norm[1]>norm[0]*tolerance)
    assert not(weights[0]==0 and norm[0]>norm[1]*tolerance)
    return Tn

def RepMat_Z2(dimR_leg1:'tuple[int]',dimR_leg2:'tuple[int]'):
    "returns P,dimR_P with indices 'aij'"
    dimR_leg_a=(dimR_leg1[0]*dimR_leg2[0]+dimR_leg1[1]*dimR_leg2[1],dimR_leg1[0]*dimR_leg2[1]+dimR_leg1[1]*dimR_leg2[0])
    dimR_P=(dimR_leg_a,dimR_leg1,dimR_leg2)
    P=torch.zeros(tuple(sum(dimR_leg) for dimR_leg in dimR_P))
    counter=0
    for i in range(dimR_leg1[0]):
        for j in range(dimR_leg2[0]):
            P[counter,i,j]=1
            counter+=1
    for i in range(dimR_leg1[1]):
        for j in range(dimR_leg2[1]):
            P[counter,dimR_leg1[0]+i,dimR_leg2[0]+j]=1
            counter+=1
    for i in range(dimR_leg1[0]):
        for j in range(dimR_leg2[1]):
            P[counter,i,dimR_leg2[0]+j]=1
            counter+=1
    for i in range(dimR_leg1[1]):
        for j in range(dimR_leg2[0]):
            P[counter,dimR_leg1[0]+i,j]=1
            counter+=1
    return P,dimR_P # 'aij' (dimV1*dimV2,dimV1,dimV2) 

def merge_legs_Z2(T,dimR:'tuple[tuple[int]]',i,j):
    "returns T_new,dimR,P,dimR_P. P's indices 'aij'. T_new=P:T"
    assert check_dimR(T,dimR)
    assert i!=j
    if i>j: i,j=j,i
    P,dimR_P=RepMat_Z2(dimR[i],dimR[j]) # aij
    T_idx=tuple(range(len(T.shape)))
    T_idx_new=T_idx[:i]+(-1,)+T_idx[i+1:j]+T_idx[j+1:]
    dimR_new=dimR[:i]+ dimR_P[0] +dimR[i+1:j]+dimR[j+1:]
    T_new=contract(T,T_idx,P,(-1,i,j,),T_idx_new)
    return T_new,dimR_new,P,dimR_P


def truncate_svd_result_Z2(u,s,vh,dimR_u:'tuple[tuple[int]]',max_dim):
    start,mid,end=0,dimR_u[0][0],dimR_u[0][0]+dimR_u[0][1]
    idx=torch.argsort(s,descending=True)
    idx=idx[:max_dim]
    idx1,idx2=idx[idx<mid],idx[idx>=mid]
    dimR_u_new=(dimR_u[0],(len(idx1),len(idx2)))
    idx=torch.cat((idx1,idx2))
    u_new,s_new,vh_new=u[:,idx],s[idx],vh[idx,:]
    return u_new,s_new,vh_new,dimR_u_new

def svd_Z2(M,dimR:'tuple[tuple[int]]',max_dim=None,**args):
    "returns u,s,vh,dimR_u"
    assert check_dimR(M,dimR)
    assert len(M.shape)==2 and dimR[0]==dimR[1], 'M is not a matrix'
    assert Z2_sector_norm(M,dimR)[1]==0, 'M is not Z2 symmetric'
    start,mid,end=0,dimR[0][0],dimR[0][0]+dimR[0][1]
    u1,s1,vh1=svd(M[:mid,:mid],**args)
    u2,s2,vh2=svd(M[mid:,mid:],**args)
    u=torch.block_diag(u1,u2)
    s=torch.cat((s1,s2))
    vh=torch.block_diag(vh1,vh2)
    dimR_u=(dimR[0],dimR[0])
    if max_dim is not None:
        return truncate_svd_result_Z2(u,s,vh,dimR_u,max_dim=max_dim)
    else:
        return u,s,vh,dimR_u










def default_dimR(T)->'tuple[tuple[int]]':
    return tuple((d,) for d in T.shape)


def check_dimR(T,dimR:'tuple[tuple[int]]'):
    return len(T.shape)==len(dimR) and all(i==sum(j) for i,j in zip(T.shape,dimR))