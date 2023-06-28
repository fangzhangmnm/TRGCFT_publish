import torch
import numpy as np
from opt_einsum import contract
from scipy.special import comb

from torch.linalg import matrix_power as mpow

Models={}

def _register_model(cls):
    Models[cls.__name__]=cls
    return cls

class TNModel:
    def __init__(self,params):
        params={k:(params[k] if k in params else v) for k,v in self.get_default_params().items()}
        self.params={k:torch.as_tensor(v).type(torch.get_default_dtype()) for k,v in params.items()}
    def get_observable(self,name):
        rtval=self.get_observable_moments(name,n=1)
        return rtval[0][1],rtval[1]
    def get_observable_moments(self,name,n):
        raise NotImplementedError()
    def get_T0(self):
        raise NotImplementedError()
    
@_register_model
class Ising2D(TNModel):
    @staticmethod 
    def get_default_params():
        return {'beta':np.log(1+2**.5)/2,'h':0}#0.44068679350977147
    def __init__(self,params={}):
        super().__init__(params)
        self.spacial_dim=2
    def get_observable_moments(self,name,n):
        if name=='magnetization':
            return [self.get_SZT0(moment=i) for i in range(n+1)],False
    def get_T0(self):
        return self.get_T(torch.tensor([1,1]))
    def get_SZT0(self,moment=1):
        return self.get_T(torch.tensor([1,(-1)**moment]))
    def get_T(self,op):
        beta,h=self.params['beta'],self.params['h']
        a=torch.sqrt(torch.cosh(beta))
        b=torch.sqrt(torch.sinh(beta))
        W=torch.stack([torch.stack([a,b]),torch.stack([a,-b])])
        sz=torch.stack([torch.exp(beta*h),torch.exp(-beta*h)])*op
        return contract('Ai,Aj,Ak,Al,A->ijkl',W,W,W,W,sz)#UDLR

    
@_register_model
class Ising3D(TNModel):
    @staticmethod 
    def get_default_params():
        return {'beta':0.2216544,'h':0}
    def __init__(self,params={}):
        super().__init__(params)
        self.spacial_dim=3
    def get_observable_moments(self,name,n):
        if name=='magnetization':
            return [self.get_SZT0(moment=i) for i in range(n+1)],False
    def get_T0(self):
        return self.get_T(torch.tensor([1,1]))
    def get_SZT0(self,moment=1):
        return self.get_T(torch.tensor([1,(-1)**moment]))
    def get_T(self,op):
        beta,h=self.params['beta'],self.params['h']
        a=torch.sqrt(torch.cosh(beta))
        b=torch.sqrt(torch.sinh(beta))
        W=torch.stack([torch.stack([a,b]),torch.stack([a,-b])])
        sz=torch.stack([torch.exp(beta*h),torch.exp(-beta*h)])*op
        return contract('Ai,Aj,Ak,Al,Am,An,A->ijklmn',W,W,W,W,W,W,sz)

class AKLTModel(TNModel):
    def get_observable_moments(self,name,n):
        if name=='magnetizationX':
            return [self.get_ST0(axis=0,moment=i) for i in range(n+1)],self._aklt_checkboard_cell
        if name=='magnetizationY':
            return [self.get_ST0(axis=1,moment=i) for i in range(n+1)],self._aklt_checkboard_cell
        if name=='magnetizationZ':
            return [self.get_ST0(axis=2,moment=i) for i in range(n+1)],self._aklt_checkboard_cell

    
@_register_model
class AKLT2D(AKLTModel):
    @staticmethod 
    def get_default_params():
        return {'a1':np.sqrt(6/4),'a2':np.sqrt(6/1)}
    def __init__(self,params={}):
        super().__init__(params)
        self.spacial_dim=2
        self._aklt_checkboard_cell=True
    def get_T0(self):
        return self.get_T(get_Identity(2))
    def get_ST0(self,axis,moment=1):
        return self.get_T(mpow(get_Lxyz(j=2)[axis],moment))
    def get_T(self,op):
        projector=get_CG_no_normalization(2)
        singlet=get_Singlet()
        ac0,ac1,ac2=torch.tensor(1),self.params['a1'],self.params['a2']
        deform=torch.stack([ac2,ac1,ac0,ac1,ac2])
        node=contract('aIjKl,iI,kK,a->aijkl',projector,singlet,singlet,deform)
        T=contract('aijkl,AIJKL,aA->iIjJkKlL',node,node,op).reshape(4,4,4,4)#UDLR
        r=get_AKLT_Rep_Isometry()
        T=contract('ijkl,Ii,Jj,Kk,Ll->IJKL',T,r,r.conj(),r,r.conj())
        return T
    
@_register_model
class AKLT3D(AKLTModel):
    @staticmethod 
    def get_default_params():
        # 1.154 1.826 4.472
        return {'a1':np.sqrt(20/15),'a2':np.sqrt(20/6),'a3':np.sqrt(20/1)}
    def __init__(self,params={}):
        super().__init__(params)
        self.spacial_dim=3
        self._aklt_checkboard_cell=True
    def get_T0(self):
        return self.get_T(get_Identity(j=3))
    def get_ST0(self,axis,moment=1):
        return self.get_T(mpow(get_Lxyz(j=3)[axis],moment))
    def get_T(self,op):
        projector=get_CG_no_normalization(3)
        singlet=get_Singlet()
        ac0,ac1,ac2,ac3=torch.tensor(1),self.params['a1'],self.params['a2'],self.params['a3']
        deform=torch.stack([ac3,ac2,ac1,ac0,ac1,ac2,ac3])
        node=contract('aIjKlMn,iI,kK,mM,a->aijklmn',projector,singlet,singlet,singlet,deform)
        T=contract('aijklmn,AIJKLMN,aA->iIjJkKlLmMnN',node,node,op).reshape(4,4,4,4,4,4)#UDLRFB
        r=get_AKLT_Rep_Isometry()
        T=contract('ijklmn,Ii,Jj,Kk,Ll,Mm,Nn->IJKLMN',T,r,r.conj(),r,r.conj(),r,r.conj())
        return T
    



@_register_model
class AKLTDiamond(AKLTModel):
    @staticmethod 
    def get_default_params():
        return {'a1':np.sqrt(6/4),'a2':np.sqrt(6/1)}
    def __init__(self,params={}):
        super().__init__(params)
        self.spacial_dim=3
        self._aklt_checkboard_cell=False # two sites are in one unit cell
    def get_T0(self):
        IdA=get_Identity(2)
        ids=[IdA]*2
        return self.get_T(ids)
    def get_ST0(self,axis,moment=1,weights=[1,0]):
        IdA,opA=get_Identity(j=2),mpow(get_Lxyz(j=2)[axis],moment)
        ids,ops=[IdA]*2,[opA]*2
        rtval=0
        for i in range(2):
            masked_ops=ids.copy();masked_ops[i]=ops[i]
            rtval+=weights[i]*self.get_T(ops)
        return rtval
    def get_T(self,ops):
        projectorA=get_CG_no_normalization(2)
        singlet=get_Singlet()
        ac0,ac1,ac2=torch.tensor(1),self.params['a1'],self.params['a2']
        deformA=torch.stack([ac2,ac1,ac0,ac1,ac2])
        node=contract('axIKM,bXjln,iI,kK,mM,xX,a,b->abijklmn',
                      *([projectorA]*2+[singlet]*4+[deformA]*2))
        T=contract('abijklmn,ABIJKLMN,aA,bB->iIjJkKlLmMnN',
                   *([node,node]+ops)).reshape(4,4,4,4,4,4)#UDLRFB
        r=get_AKLT_Rep_Isometry()
        T=contract('ijklmn,Ii,Jj,Kk,Ll,Mm,Nn->IJKLMN',T,r,r.conj(),r,r.conj(),r,r.conj())
        return T
    

@_register_model
class AKLTSinglyDecoratedDiamond(AKLTModel):
    @staticmethod 
    def get_default_params():
        return {'a1':np.sqrt(6/4),'a2':np.sqrt(6/1),'b1':np.sqrt(2/1)}
    def __init__(self,params={}):
        super().__init__(params)
        self.spacial_dim=3
        self._aklt_checkboard_cell=False
    def get_T0(self):
        IdA,IdB=get_Identity(j=2),get_Identity(j=1)
        ids=[IdA]*2+[IdB]*4
        return self.get_T(ids)
    def get_ST0(self,axis,moment=1,weights=[1,0,0,0,0,0]):
        IdA,IdB,opA,opB=get_Identity(j=2),get_Identity(j=1),mpow(get_Lxyz(j=2)[axis],moment),mpow(get_Lxyz(j=1)[axis],moment)
        ids,ops=[IdA]*2+[IdB]*4,[opA]*2+[opB]*4
        rtval=0
        for i in range(6):
            masked_ops=ids.copy();masked_ops[i]=ops[i]
            rtval+=weights[i]*self.get_T(ops)
        return rtval
    def get_T(self,ops):
        projectorA=get_CG_no_normalization(2)
        projectorB=get_CG_no_normalization(1)
        singlet=get_Singlet()
        ac0,ac1,ac2=torch.tensor(1),self.params['a1'],self.params['a2']
        bc0,bc1=torch.tensor(1),self.params['b1']
        deformA=torch.stack([ac2,ac1,ac0,ac1,ac2])
        deformB=torch.stack([bc1,bc0,bc1])
        node=contract('axUVW,bYjln,cXy,dIv,eKu,fMw,iI,kK,mM,uU,vV,wW,xX,yY,a,b,c,d,e,f->abcdefijklmn',
                      *([projectorA]*2+[projectorB]*4+[singlet]*8+[deformA]*2+[deformB]*4))
        T=contract('abcdefijklmn,ABCDEFIJKLMN,aA,bB,cC,dD,eE,fF->iIjJkKlLmMnN',
                   *([node,node]+ops)).reshape(4,4,4,4,4,4)#UDLRFB
        r=get_AKLT_Rep_Isometry()
        T=contract('ijklmn,Ii,Jj,Kk,Ll,Mm,Nn->IJKLMN',T,r,r.conj(),r,r.conj(),r,r.conj())
        return T
    
    
@_register_model
class AKLTHoneycomb(AKLTModel):
    @staticmethod 
    def get_default_params():
        return {'a32':np.sqrt(3/1)}
    def __init__(self,params={}):
        super().__init__(params)
        self.spacial_dim=2
        self._aklt_checkboard_cell=False # two sites are in one unit cell
    def get_T0(self):
        Id=get_Identity(3/2)
        ids=[Id]*2
        return self.get_T(ids)
    def get_ST0(self,axis,moment=1,weights=[1,0]):
        Id,op=get_Identity(j=3/2),mpow(get_Lxyz(j=3/2)[axis],moment)
        ids,ops=[Id]*2,[op]*2
        rtval=0
        for i in range(2):
            masked_ops=ids.copy();masked_ops[i]=ops[i]
            rtval+=weights[i]*self.get_T(ops)
        return rtval
    def get_T(self,ops):
        projector=get_CG_no_normalization(3/2)
        singlet=get_Singlet()
        ac12,ac32=torch.tensor(1),self.params['a32']
        deform=torch.stack([ac32,ac12,ac12,ac32])
        node=contract('aIKx,bjlX,iI,kK,xX,a,b->abijkl',projector,projector,singlet,singlet,singlet,deform,deform)
        T=contract('abijkl,ABIJKL,aA,bB->iIjJkKlL',node,node,ops[0],ops[1]).reshape(4,4,4,4)#UDLR
        r=get_AKLT_Rep_Isometry()
        T=contract('ijkl,Ii,Jj,Kk,Ll->IJKL',T,r,r.conj(),r,r.conj())
        return T
    
    
    
        
    
    

# WARNING! DO NOT SIMPLY UNCOMMENT THE BELOW CODE. I CHANGED A LOT ON THE ABOVE CODE BUT NOT UPDATE THE BELOW CODE YET.
# WARNING! DO NOT SIMPLY UNCOMMENT THE BELOW CODE. I CHANGED A LOT ON THE ABOVE CODE BUT NOT UPDATE THE BELOW CODE YET.
# WARNING! DO NOT SIMPLY UNCOMMENT THE BELOW CODE. I CHANGED A LOT ON THE ABOVE CODE BUT NOT UPDATE THE BELOW CODE YET.
# WARNING! DO NOT SIMPLY UNCOMMENT THE BELOW CODE. I CHANGED A LOT ON THE ABOVE CODE BUT NOT UPDATE THE BELOW CODE YET.
# WARNING! DO NOT SIMPLY UNCOMMENT THE BELOW CODE. I CHANGED A LOT ON THE ABOVE CODE BUT NOT UPDATE THE BELOW CODE YET.
# some changes: checkerboard FFT->TTT

    
# @_register_model
# class AKLT2DStrange(TNModel):
#     @staticmethod 
#     def get_default_params():
#         return {'a1':np.sqrt(6/4),'a2':np.sqrt(6/1)}
#     def __init__(self,params={}):
#         super().__init__(params)
#         self.spacial_dim=2
        
#     def get_dimR(self,Z2=True):
#         # TODO ?????
#         return ((2,0),)*self.spacial_dim

        
#     def get_T(self,op):
#         projector=get_CG_no_normalization(2)
#         singlet=get_Singlet()
#         ac0,ac1,ac2=torch.tensor(1),self.params['a1'],self.params['a2']
#         deform=torch.stack([ac2,ac1,ac0,ac1,ac2])
#         AKLTnode=contract('aIjKl,iI,kK,a->aijkl',projector,singlet,singlet,deform)
#         productStateNode=torch.tensor([0.,0.,1.,0.,0.])
#         T=contract('aijkl,A,aA->ijkl',AKLTnode,productStateNode,op).reshape(2,2,2,2)#UDLR
#         return T
        
#     def get_T0(self):
#         return self.get_T(get_Identity(2))
    
#     def get_ST0(self,axis):
#         # checkerboard=False,False,True
#         return self.get_T(get_Lxyz(j=2)[axis])
    



    
    


def get_CG_no_normalization(j):
    n=int(2*j)
    if n==0:
        return torch.eye(1)
    CG=torch.zeros((n+1,)+(2,)*n)
    for i in range(2**n):
        indices=tuple(map(int,bin(i)[2:].zfill(n)))
        m=sum(indices)
        CG[(m,)+indices]=1
    return CG
def get_CG(j):
    n=int(2*j)
    if n==0:
        return torch.eye(1)
    CG=torch.zeros((n+1,)+(2,)*n)
    for i in range(2**n):
        indices=tuple(map(int,bin(i)[2:].zfill(n)))
        m=sum(indices)
        CG[(m,)+indices]=1/np.sqrt(comb(n,m))
    return CG
def get_Singlet():
    return torch.tensor([[0,1.],[-1.,0]])
def get_Lxyz(j):
    n=int(2*j+1)
    Lz=torch.zeros((n,n))
    for i in range(n):
        m=i-j
        Lz[i,i]=m
    Lp=torch.zeros((n,n))
    for i in range(n-1):
        m=i-j
        Lp[i+1,i]=np.sqrt(j*(j+1)-m*(m+1))
    Lm=Lp.T
    Lx=(Lp+Lm)/2
    iLy=(Lp-Lm)/2
    return Lx,iLy,Lz
def get_Identity(j):
    n=int(2*j+1)
    return torch.eye(n)

def get_AKLT_Rep_Isometry():
    return torch.tensor([[1.,0.,0.,0.],[0.,0.,0.,1.],[0.,np.sqrt(.5),np.sqrt(.5),0.],[0.,np.sqrt(.5),-np.sqrt(.5),0.]]).type(torch.get_default_dtype())

