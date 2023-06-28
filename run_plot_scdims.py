if __name__ != '__main__':
    assert False, 'This file is not meant to be imported'

import argparse
parser = argparse.ArgumentParser()

#def show_scaling_dimensions(Ts,loop_length=2,num_scaling_dims=8,volume_scaling=2,is_HOTRG=False,reference_scaling_dimensions=None, reference_center_charge=None,filename=None):

parser.add_argument('--filename', type=str, required=True) # data/tnr_X16_L10
parser.add_argument('--tensor_path', type=str, required=True) # data/tnr_X16_tensors.pkl
parser.add_argument('--loop_length', type=int, default=2)
parser.add_argument('--is_HOTRG', action='store_true')
parser.add_argument('--num_scaling_dims', type=int, default=16)

parser.add_argument('--version', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')


args = parser.parse_args()
options=vars(args)


print('loading library...')
from opt_einsum import contract # idk why but its required to avoid bug in contract with numpy arrays
import torch
import numpy as np
torch.set_default_tensor_type(torch.cuda.DoubleTensor)
if options['device']=='cpu':
    torch.set_default_tensor_type(torch.DoubleTensor)
else:  
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
device=torch.device(options['device'])
torch.cuda.set_device(device)

# from ScalingDimensions import show_scaling_dimensions,show_diff,show_effective_rank


from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from HOTRG import cg_tensor_norm
from transfer_matrix import (
    fix_normalize,
    get_transfer_matrix_spectrum,
    get_transfer_matrix_scaling,
    get_central_charge_from_spectrum,
    get_scaling_dimensions_from_spectrum,
)

def show_diff(Ts,stride=1,filename=None):
    curve=[]
    for i,A in tqdm([*enumerate(Ts)]):
        newRow={'layer':i}
        if i-stride>=0 and A.shape==Ts[i-stride].shape:
            newRow['diff']=((Ts[i]-Ts[i-stride]).norm()/Ts[i].norm()).item()
        curve.append(newRow)
    curve=pd.DataFrame(curve)
    plt.plot(curve['layer'],curve['diff'],'.-',color='black',label='$|T\'-T|$')
    plt.xlabel('RG Step')
    plt.ylabel('$|T\'-T|/|T|$')
    plt.yscale('log')
    plt.ylim((1e-7,2))
    plt.show()
    if filename is not None:
        plt.savefig(filename+'_diff.png')
        print('saved to',filename+'_diff.png')
        plt.close()
    return curve


def NWSE(T):
    return contract('nswe->nwse',T).reshape(T.shape[0]*T.shape[2],-1)
def NESW(T):
    return contract('nswe->nesw',T).reshape(T.shape[0]*T.shape[2],-1)

def effective_rank(M):
    assert len(M.shape)==2
    u,s,vh=torch.linalg.svd(M)
    s=s[s>0]
    p=s/torch.sum(s)
    entropy=-torch.sum(p*torch.log(p))
    return torch.exp(entropy)

def show_effective_rank(Ts,filename=None):
    curve=[]

    for i,A in tqdm([*enumerate(Ts)]):
        _,s,_=torch.linalg.svd(NWSE(A))
        s=s/s[0]
        s=s.cpu().numpy()
        if(s.shape[0]<30):
            s=np.pad(s,(0,30-s.shape[0]))
        else:
            s=s[:30]
        er=effective_rank(NWSE(A)).item()
        er1=effective_rank(NESW(A)).item()
        newRow={'layer':i,'entanglement_spectrum':s,'effective_rank_nwse':er,'effective_rank_nesw':er1}
        curve.append(newRow)
    curve=pd.DataFrame(curve)

    ss=np.array(curve['entanglement_spectrum'].tolist())
    ee=curve['effective_rank_nwse'].tolist()
    iii=curve['layer']
    for sss in ss.T:
        plt.plot(iii,sss,'-k')
    plt.title(f'')
    plt.xlabel('RG Step')
    plt.ylabel('normalized eigenvalues')
    plt.show()
    if filename is not None:
        plt.savefig(filename+'_eigsd.png')
        print('saved to',filename+'_eigsd.png')
        plt.close()

    plt.plot(iii,ee,'-k',label='nwse')
    plt.ylabel('effective rank')
    plt.show()
    if filename is not None:
        plt.savefig(filename+'_rankd.png')
        print('saved to',filename+'_rankd.png')
        plt.close()
    return curve

def show_scaling_dimensions(Ts,loop_length=2,num_scaling_dims=8,volume_scaling=2,is_HOTRG=False,reference_scaling_dimensions=None, reference_center_charge=None,filename=None,display=True,stride=1):
    curve=[]
    def pad(v):
        return np.pad(v,(0,num_scaling_dims))[:num_scaling_dims]
    spacial_dim=len(Ts[0].shape)//2
    norms=list(map(cg_tensor_norm,Ts))
    for iLayer,A in tqdm([*enumerate(Ts)]):
        if iLayer%stride!=0:
            continue
        A=fix_normalize(A,is_HOTRG=is_HOTRG,volume_scaling=volume_scaling,norms=norms[:iLayer+1])
        if spacial_dim==2:
            if is_HOTRG:
                aspect=[loop_length,loop_length*2][iLayer%2]
                tensor_block_height=[1,.5][iLayer%2]
            else:
                aspect=1
                tensor_block_height=1
            
        else:
            raise NotImplementedError
        spectrum,_=get_transfer_matrix_spectrum(A,n=loop_length,k=num_scaling_dims+1)
        scaling=get_transfer_matrix_scaling(A,n=loop_length,tensor_block_height=tensor_block_height)

        center_charge=get_central_charge_from_spectrum(spectrum,scaling=scaling).item()
        scaling_dimensions=pad(get_scaling_dimensions_from_spectrum(spectrum,scaling=scaling).tolist())
        spectrum=pad(spectrum.tolist())
        
        
        newRow={'layer':iLayer,
                'center_charge':center_charge,
                'scaling_dimensions':scaling_dimensions,
                'eigs':spectrum,
                'norm':norms[iLayer]}
        curve.append(newRow)

    curve=pd.DataFrame(curve)
    if display or filename is not None:
        eigs=np.array(curve['eigs'].tolist()).T
        for eig in eigs:
            plt.plot(curve['layer'],eig,'.-',color='black')
        plt.xlabel('RG Step')
        plt.ylabel('eigenvalues of transfer matrix')
        plt.ylim([0,1])
        plt.show()
        if filename is not None:
            plt.savefig(filename+'_eigs.png')
            print('saved to',filename+'_eigs.png')
            plt.close()
        
        sdsds=np.array(curve['scaling_dimensions'].tolist()).T
        if reference_scaling_dimensions is not None:
            for sdsd in reference_scaling_dimensions:
                plt.plot(curve['layer'],np.ones_like(curve['layer'])*sdsd,'-',color='lightgrey')
            plt.ylim([0,max(reference_scaling_dimensions)*1.1])
        else:
            plt.ylim([np.average(sdsds[-1])*-.1,np.average(sdsds[-1])*1.5])

        for sdsd in sdsds:
            plt.plot(curve['layer'],sdsd,'.-',color='black')
        plt.xlabel('RG Step')
        plt.ylabel('scaling dimensions')
        plt.show()
        if filename is not None:
            plt.savefig(filename+'_scDim.png')
            print('saved to',filename+'_scDim.png')
            plt.close()
        
        if reference_center_charge is not None:
            plt.plot(curve['layer'],np.ones_like(curve['layer'])*reference_center_charge,'-',color='lightgrey')
            plt.ylim([0,reference_center_charge*2])
        else:
            avg=np.average(curve['center_charge'])
            #nan or inf
            if np.isnan(avg) or np.isinf(avg):
                avg=.5
                 
            plt.ylim([avg*-.1,avg*1.5])
        plt.plot(curve['layer'],curve['center_charge'],'.-',color='black')
        plt.xlabel('RG Step')
        plt.ylabel('central charge')
        plt.show()
        if filename is not None:
            plt.savefig(filename+'_c.png')
            print('saved to',filename+'_c.png')
            plt.close()
    
    return curve














from HOTRG import HOTRGLayer

print('loading tensors...')
loaded=torch.load(options['tensor_path'],map_location=device)
options1, params, layers, Ts, logTotals = loaded['options'], loaded['params'], loaded['layers'], loaded['Ts'], loaded['logTotals']

if isinstance(layers[0],HOTRGLayer):
    assert options['is_HOTRG']==True

print(options)
print(options1)

reference_scaling_dimensions=[0,0.125,1,1.125,2,2.125,3,3.125,4,4.125,5,5.125]
reference_central_charge=.5

stride=2 if options['is_HOTRG'] else 1
diff_curve=show_diff(Ts,stride=stride,filename=options['filename'])
effective_rank_curve=show_effective_rank(Ts,filename=options['filename'])
scaling_dimensions_curve=show_scaling_dimensions(Ts,
                        loop_length=options['loop_length'],
                        num_scaling_dims=options['num_scaling_dims'],
                        is_HOTRG=options['is_HOTRG'],
                        filename=options['filename'],
                        reference_scaling_dimensions=reference_scaling_dimensions,
                        reference_center_charge=reference_central_charge)
torch.save((diff_curve,effective_rank_curve,scaling_dimensions_curve),options['filename']+'_curves.pth')
print('saved to',options['filename']+'_curves.pth')