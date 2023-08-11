import torch
from tqdm.auto import tqdm as tqdm
from opt_einsum import contract
import torch.utils.checkpoint
import itertools as itt
import functools
from dataclasses import dataclass
import math
import numpy as np
from torch.linalg import svd

#======================== Z2 =================================

def RepDim(dimV1R1,dimV1R2,dimV2R1,dimV2R2):
    return (dimV1R1*dimV2R1+dimV1R2*dimV2R2,dimV1R1*dimV2R2+dimV1R2*dimV2R1)

def RepMat(dimV1R1,dimV1R2,dimV2R1,dimV2R2):
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

def _RepMat(a,b):
    return RepMat(a,b,a,b)

def Z2_sectors(T,dimR:'tuple[tuple[int]]'):
    if len(T.shape)==2*len(dimR): dimR=tuple(d for d in dimR for _ in range(2))
    assert len(T.shape)==len(dimR) and all(i==sum(j) for i,j in zip(T.shape,dimR))
    for sector in itt.product(range(2),repeat=len(dimR)):
        begin=[sum(dimR[leg][:rep]) for leg,rep in enumerate(sector)]
        end=[sum(dimR[leg][:rep+1]) for leg,rep in enumerate(sector)]
        slices=[slice(b,e) for b,e in zip(begin,end)]
        yield sector,slices

def Z2_sector_norm(T,dimR:'tuple[tuple[int]]'):
    if len(T.shape)==2*len(dimR): dimR=tuple(d for d in dimR for _ in range(2))
    if len(T.shape)==2*len(dimR): dimR=(d for d in dimR for _ in range(2))
    assert len(T.shape)==len(dimR) and all(i==sum(j) for i,j in zip(T.shape,dimR))
    sqrnorm=torch.zeros(2)
    for sector,slices in Z2_sectors(T,dimR):
        sqrnorm[sum(sector)%2]+=T[slices].norm()**2
    return sqrnorm**.5

def project_Z2(T,dimR:'tuple[tuple[int]]',weights=[1,0],tolerance=float('inf')):
    if len(T.shape)==2*len(dimR): dimR=tuple(d for d in dimR for _ in range(2))
    if len(T.shape)==2*len(dimR): dimR=(d for d in dimR for _ in range(2))
    assert len(T.shape)==len(dimR) and all(i==sum(j) for i,j in zip(T.shape,dimR))
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
    if len(T.shape)==2*len(dimR): dimR=tuple(d for d in dimR for _ in range(2))
    if len(T.shape)==2*len(dimR): dimR=(d for d in dimR for _ in range(2))
    assert len(T.shape)==len(dimR) and all(i==sum(j) for i,j in zip(T.shape,dimR))
    assert i!=j
    if i>j: i,j=j,i
    P=RepMat(dimR[i][0],dimR[i][1],dimR[j][0],dimR[j][1])
    T_idx=tuple(range(len(T.shape)))
    T_idx_new=T_idx[:i]+(-1,)+T_idx[i+1:j]+T_idx[j+1:]
    T=contract(T,T_idx,P,(-1,i,j,),T_idx_new)
    dimRnew=dimR[:i]+ (RepDim(dimR[i][0],dimR[i][1],dimR[j][0],dimR[j][1]),) +dimR[i+1:j]+dimR[j+1:]
    return T,dimRnew

#============================= Forward Layers ======================================

def is_isometry(g):
    return torch.isclose(g@g.T.conj(),torch.eye(g.shape[0])).all()


@dataclass
class HOTRGLayer:
    tensor_shape:'tuple[int]'
    ww:'list[torch.Tensor]'
    dimR:'tuple[tuple[int]]'=None
    dimR_next:'tuple[tuple[int]]'=None
    gg:'list[list[torch.Tensor]]'=None
    hh:'list[list[torch.Tensor]]'=None

    def get_isometry(self,iLeg):
        #         h0
        #         g00                   
        #    /g02-Ta-g03\       0       2
        #h2-w0    |g..  w0-h3  2T3  -> 0T'1  
        #    \g12-Tb-g13/       1       3
        #         g11                      
        #         h1
        iAxis=iLeg//2
        if iAxis==0: # first pair of virtual leg
            w=torch.eye(self.tensor_shape[iLeg])
            if self.gg:
                w=self.gg[iLeg][iLeg]@w
            if self.hh:
                w=self.hh[iLeg]@w
        else:
            w=self.ww[iAxis-1]
            if self.dimR:
                P=_RepMat(self.dimR[iAxis][0],self.dimR[iAxis][1])
                w=contract('ab,bij->aij',w,P)
            else:
                w=w.reshape(-1,self.tensor_shape[iLeg],self.tensor_shape[iLeg])
            if iLeg%2==1:
                w=w.conj()
            if self.gg:
                w=contract('aij,iI,jJ->aIJ',w,self.gg[0][iLeg],self.gg[1][iLeg])
            if self.hh:
                w=contract('aij,Aa->Aij',w,self.hh[iLeg])
        return w
    def get_insertion(self):
        if self.gg:
            return self.gg[0][1].T@self.gg[1][0]
        else:
            return torch.eye(self.tensor_shape[0])
        

def _forward_layer(Ta,Tb,layer:HOTRGLayer):
    assert layer.tensor_shape==Ta.shape and layer.tensor_shape==Tb.shape
    isometries=[layer.get_isometry(i) for i in range(len(layer.tensor_shape))]
    insertion=layer.get_insertion()
    eq={4:'ijkl,Jmno,jJ,xi,ym,akn,blo->abxy',
        6:'ijklmn,Jopqrs,jJ,xi,yo,akp,blq,cmr,dns->abcdxy',
        }[len(layer.tensor_shape)]
    T=contract(eq,Ta,Tb,insertion,*isometries)
    return T
    
def _checkpoint(function,args,args1,use_checkpoint=True):
    if use_checkpoint and any(x.requires_grad for x in args):
        def wrapper(*args):
            return function(*args,**args1)
        return torch.utils.checkpoint.checkpoint(wrapper,*args)
    else:
        return function(*args,**args1)
    
def forward_layer(Ta,Tb,layer:HOTRGLayer,use_checkpoint=False)->torch.Tensor:
    #_forward_layer={4:_forward_layer_2D,6:_forward_layer_3D}[len(Ta.shape)]
    return _checkpoint(_forward_layer,[Ta,Tb],{'layer':layer},use_checkpoint=use_checkpoint)

def cg_tensor_norm(T):
    contract_path={4:'iijj->',6:'iijjkk->'}[len(T.shape)]
    norm=contract(contract_path,T).norm()
    if norm<1e-6*T.norm():#fallback
        # print('cg_tensor_norm: fallback',T.shape)
        norm=T.norm()
    # norm=T.norm()
    #print(norm)
    return norm
    
def to_unitary(g):
    u,s,vh=svd(g)
    return u@vh
    
def forward_tensor(T0,layers:'list[HOTRGLayer]',use_checkpoint=False,return_layers=False):
    T,logTotal=T0,0
    if return_layers: 
        Ts,logTotals=[T],[0]
    for layer in tqdm(layers,leave=False):
        norm=cg_tensor_norm(T)
        T=T/norm
        logTotal=2*(logTotal+norm.log())
        T=forward_layer(T,T,layer=layer,use_checkpoint=use_checkpoint)
        if return_layers: 
            Ts.append(T);logTotals.append(logTotal)
    return (Ts,logTotals) if return_layers else (T,logTotal)

def forward_observable_tensor(T0,T0_op,layers:'list[HOTRGLayer]',
        start_layer=0,checkerboard=False,use_checkpoint=False,return_layers=False,cached_Ts=None):
    spacial_dim=len(T0.shape)//2
    T,logTotal=forward_tensor(T0,layers=layers[:start_layer],use_checkpoint=use_checkpoint,return_layers=return_layers)
    T_op=T0_op
    if return_layers:
        Ts,T,logTotals,logTotal=T,T[-1],logTotal,logTotal[-1]
        T_ops=[None]*start_layer+[T_op]
    for ilayer,layer in tqdm(list(enumerate(layers))[start_layer:],leave=False):
        norm=cg_tensor_norm(T)
        T,T_op=T/norm,T_op/norm
        logTotal=2*(logTotal+norm.log())
        if cached_Ts:
            T1=cached_Ts[ilayer+1]
        else:
            T1=forward_layer(T,T,layer=layer,use_checkpoint=use_checkpoint)
        #with BypassGilt(False,True):
        T2=forward_layer(T,T_op,layer=layer,use_checkpoint=use_checkpoint)
        #with BypassGilt(True,False):
        T3=forward_layer(T_op,T,layer=layer,use_checkpoint=use_checkpoint)
        T3=-T3 if (checkerboard and ilayer<spacial_dim) else T3
        T,T_op=T1,(T2+T3)/2
        if return_layers:
            Ts.append(T);T_ops.append(T_op);logTotals.append(logTotal)
    return (Ts,T_ops,logTotals) if return_layers else (T,T_op,logTotal)

    
def forward_observalbe_tensor_moments(T0_moments:'list[torch.Tensor]',layers:'list[HOTRGLayer]',
        checkerboard=False,use_checkpoint=False,return_layers=False,cached_Ts=None):\
    # -T'[OO]- = -T[OO]-T[1]- + 2 -T[O]-T[O]- + -T[1]-T[OO]-      
    spacial_dim=len(T0_moments[0].shape)//2
    logTotal=0
    Tms=T0_moments.copy()
    if return_layers:
        Tmss,logTotals=[Tms],[logTotal]
    for iLayer,layer in tqdm(list(enumerate(layers)),leave=False):
        norm=cg_tensor_norm(Tms[0])
        logTotal=2*(logTotal+norm.log())
        Tms=[x/norm for x in Tms]
        Tms1=[0]*len(Tms)
        # print('layer',iLayer)
        for orderA in range(len(Tms)):
            for orderB in range(len(Tms)):
                if orderA+orderB<len(Tms1):
                    if orderA+orderB==0 and cached_Ts:
                        Tms1[orderA+orderB]=cached_Ts[iLayer+1]
                    else:
                        sign=-1 if (checkerboard and iLayer<spacial_dim and orderB%2==1) else 1
                        coeff=math.comb(orderA+orderB,orderB)/2**(orderA+orderB)
                        # print('<s^{}>^({})+={:.3f}<s^{}|s^{}>^({})'.format(orderA+orderB,iLayer+1,sign*coeff,orderA,orderB,iLayer))
                        contracted=forward_layer(Tms[orderA],Tms[orderB],layer=layer,use_checkpoint=use_checkpoint)
                        Tms1[orderA+orderB]=Tms1[orderA+orderB]+sign*coeff*contracted
        Tms=Tms1
        if return_layers:
            Tmss.append(Tms);logTotals.append(logTotal)
    return (Tmss,logTotals) if return_layers else (Tms,logTotal)
    
def get_lattice_size(nLayers,spacial_dim):
    return tuple(2**(nLayers//spacial_dim+(1 if i<nLayers%spacial_dim else 0)) for i in range(spacial_dim))

def get_dist_2D(x,y):
    return (x**2+y**2)**.5

def get_dist_torus_2D(x,y,lattice_size):
    # modulus but return positive numers
    x,y=x%lattice_size[0],y%lattice_size[1]
    d1=x**2+y**2
    d2=(lattice_size[0]-x)**2+y**2
    d3=x**2+(lattice_size[1]-y)**2
    d4=(lattice_size[0]-x)**2+(lattice_size[1]-y)**2
    return functools.reduce(np.minimum,[d1,d2,d3,d4])**.5

def forward_coordinate(coords):
    return coords[1:]+(coords[0]//2,)




def forward_observable_tensors(T0,T0_ops:list,positions:'list[tuple[int]]',
        layers:'list[HOTRGLayer]',checkerboard=False,use_checkpoint=False,cached_Ts=None,user_tqdm=True):
    spacial_dim=len(T0.shape)//2
    nLayers=len(layers)
    lattice_size=get_lattice_size(nLayers,spacial_dim=spacial_dim)
    assert all(isinstance(c,int) and 0<=c and c<s for coords in positions for c,s in zip(coords,lattice_size)),"coordinates must be integers in the range [0,lattice_size)\n"+str(positions)+" "+str(lattice_size)
    assert all(positions[i]!=positions[j] for i,j in itt.combinations(range(len(positions)),2))
    assert len(positions)==len(T0_ops)
    T,T_ops,logTotal=T0,T0_ops.copy(),0
    _tqdm=tqdm if user_tqdm else lambda x,leave:x
    for ilayer,layer in _tqdm(list(enumerate(layers)),leave=False):
        norm=cg_tensor_norm(T)
        logTotal=2*(logTotal+norm.log())
        T,T_ops=T/norm,[T_op/norm for T_op in T_ops]
        # check if any two points are going to merge
        iRemoved=[]
        T_ops_new,positions_new=[],[]
        for i,j in itt.combinations(range(len(positions)),2):
            if forward_coordinate(positions[i])==forward_coordinate(positions[j]):
                i,j=(i,j) if positions[i][0]%2==0 else (j,i)
                #print(positions[i],positions[j])
                assert positions[i][0]%2==0 and positions[j][0]%2==1
                #with BypassGilt(True,True):
                T_op_new=forward_layer(T_ops[i],T_ops[j],layer,use_checkpoint=use_checkpoint)
                if checkerboard and ilayer<spacial_dim:
                    T_op_new=-T_op_new
                T_ops_new.append(T_op_new)
                positions_new.append(forward_coordinate(positions[i]))
                assert (not i in iRemoved) and (not j in iRemoved)
                iRemoved.extend([i,j])
        # forward other points with T
        for i in range(len(positions)):
            if i not in iRemoved:
                if positions[i][0]%2==0:
                    #with BypassGilt(False,True):
                    T_op_new=forward_layer(T_ops[i],T,layer,use_checkpoint=use_checkpoint)
                else:
                    #with BypassGilt(True,False):
                    T_op_new=forward_layer(T,T_ops[i],layer,use_checkpoint=use_checkpoint)
                    if checkerboard and ilayer<spacial_dim:
                        T_op_new=-T_op_new
                T_ops_new.append(T_op_new)
                positions_new.append(forward_coordinate(positions[i]))
        # forward T
        if cached_Ts:
            T_new=cached_Ts[ilayer+1]
        else:
            T_new=forward_layer(T,T,layer=layer,use_checkpoint=use_checkpoint)
        T,T_ops,positions=T_new,T_ops_new,positions_new
    if len(positions)==0:
        return T,T,logTotal
    else:
        assert len(positions)==1
        return T,T_ops[0],logTotal

    
    
def trace_tensor(T):
    eq={4:'aabb->',6:'aabbcc->'}[len(T.shape)]
    return contract(eq,T)

def trace_two_tensors(T,T1=None):
    T1=T if T1 is None else T1
    eq={4:'abcc,badd->',6:'abccdd,baeeff->'}[len(T.shape)]
    return contract(eq,T,T)
 
def reflect_tensor_axis(T):
    Ai=[2*i+j for i in range(len(T.shape)//2) for j in range(2)]
    Bi=[2*i+1-j for i in range(len(T.shape)//2) for j in range(2)]
    return contract(T,Ai,Bi)
    
def permute_tensor_axis(T):
    Ai=[*range(len(T.shape))]
    Bi=Ai[2:]+Ai[:2]
    return contract(T,Ai,Bi)
#==================

import importlib
import HOSVD,GILT,fix_gauge
importlib.reload(HOSVD)
importlib.reload(GILT)
importlib.reload(fix_gauge)

from HOSVD import HOSVD_layer
from GILT import GILT_HOTRG,GILT_options
from fix_gauge import minimal_canonical_form,fix_unitary_gauge,MCF_options
    


def HOTRG_layer(T1,T2,max_dim,options:dict={},Tref=None):
    T1old,T2old=T1,T2
    gilt_options=GILT_options(**{k[5:]:v for k,v in options.items() if k[:5]=='gilt_'})
    mcf_options=MCF_options(**{k[4:]:v for k,v in options.items() if k[:4]=='mcf_'})

    T1,T2,gg=GILT_HOTRG(T1,T2,options=gilt_options) if gilt_options.enabled else (T1,T2,None)
    
    Tn,layer=HOSVD_layer(T1,T2,max_dim=max_dim)
    layer.gg=gg
    
    Tn,hh=minimal_canonical_form(Tn,options=mcf_options)
    if Tref is not None and Tn.shape==Tref.shape:
        Tn,hh1=fix_unitary_gauge(Tn,Tref,options=mcf_options)
        hh=[h1@h for h1,h in zip(hh1,hh)]

    if hh is not None:
        hh=hh[-2:]+hh[:-2] # why?
    layer.hh=hh
    if options.get('hotrg_sanity_check',False):
        Tn1= forward_layer(T1old,T2old,layer)
        assert (Tn-Tn1).abs().max()<options.get('hotrg_sanity_check_tol',1e-7)

    return Tn,layer
    
    
def HOTRG_layers(T0,max_dim,nLayers,options:dict={}):    
    print('Generating HOTRG layers')
    spacial_dim=len(T0.shape)//2
    stride=spacial_dim
    T,logTotal=T0,0
    Ts,logTotals=[T],[0]
    layers=[]
    for iLayer in tqdm(list(range(nLayers)),leave=False):
        norm=cg_tensor_norm(T)
        T=T/norm
        logTotal=2*(logTotal+norm.log())
        
        Tref=Ts[iLayer+1-stride] if iLayer+1-stride>=0 else None
        Told=T
        T,layer=HOTRG_layer(T,T,max_dim=max_dim,Tref=Tref,options=options)

        if options.get('hotrg_sanity_check',False):
            assert ((forward_layer(Told,Told,layer)-T).norm()/T.norm()<=options.get('hotrg_sanity_check_tol',1e-7))

        layers.append(layer)
        Ts.append(T);logTotals.append(logTotal)
            
    print('HOTRG layers generated')
    return layers,Ts,logTotals

    