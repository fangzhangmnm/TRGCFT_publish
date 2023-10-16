if __name__ != '__main__':
    assert False, 'This file is not meant to be imported'

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--filename', type=str, required=True) # data/hotrg_gilt_X24_Tc
parser.add_argument('--log_filename', type=str, default=None) # data/hotrg_gilt_X24_Tc.log
parser.add_argument('--nLayers', type=int, required=True) # 60
parser.add_argument('--max_dim', type=int, required=True) # 24
parser.add_argument('--model', type=str, required=True) # 'Ising2D'
parser.add_argument('--param_name', type=str, required=True) # 'beta'
parser.add_argument('--param_min', type=float, required=True) # 0.43068679350977147
parser.add_argument('--param_max', type=float, required=True) # 0.45068679350977147
parser.add_argument('--observable_name', type=str, required=True) # 'magnetization'
parser.add_argument('--tol', type=float, default=1e-8)
parser.add_argument('--gilt_enabled', action='store_true')
parser.add_argument('--gilt_eps', type=float, default=8e-7)
parser.add_argument('--gilt_nIter', type=int, default=1)
parser.add_argument('--mcf_enabled', action='store_true')
parser.add_argument('--mcf_eps', type=float, default=1e-16)
parser.add_argument('--mcf_max_iter', type=int, default=200)
parser.add_argument('--hotrg_sanity_check', action='store_true')
parser.add_argument('--version', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--method',choices=['obs','all_norm','final_norm','logZ'],default='obs')
args = parser.parse_args()
options=vars(args)


import os
if not options['overwrite'] and os.path.exists(options['filename']):
    print('file already exists, exiting')
    exit()
os.makedirs(os.path.dirname(options['filename']),exist_ok=True)
if options['log_filename'] is not None:
    os.makedirs(os.path.dirname(options['log_filename']),exist_ok=True)
logfile=open(options['log_filename'],'w') if options['log_filename'] is not None else None
print('logging to',options['log_filename'])


def print_and_log(*args):
    print(*args)
    if logfile is not None:
        print(*args,file=logfile)
        logfile.flush()

print_and_log('Start running find_critical_temp.py with options:'+str(options))
print_and_log('loading library...')

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


from HOTRG import HOTRG_layers,trace_tensor,forward_observable_tensor,trace_two_tensors
from TNModels import Models

Model=Models[options['model']]
params=Model.get_default_params()
param_name=options['param_name']


def eval_model(params):
    model=Model(params)
    T0,(T0_op,checkerboard)=model.get_T0(),model.get_observable(options['observable_name'])
    layers,Ts,logTotals=HOTRG_layers(T0,
                        max_dim=options['max_dim'],nLayers=options['nLayers'],
                        options=options)
    Ts,T_ops,logTotals=forward_observable_tensor(T0,T0_op,
                        layers=layers,checkerboard=checkerboard,
                        return_layers=True,cached_Ts=Ts)
    T=Ts[-1]/Ts[-1].norm()
    logZ=(trace_tensor(T).log()+logTotals[-1])/2**options['nLayers']
    dNorm=torch.tensor([T.norm() for T in Ts]) # according to Lyu, this can be used to determine the phase transition
    #dNorm=T.norm() 
    obs=trace_two_tensors(T_ops[-1])/trace_two_tensors(Ts[-1])
    return T,logZ,obs,dNorm

beta_min=options['param_min']
beta_max=options['param_max']
beta_ref=Model.get_default_params()[param_name]

print_and_log('evaluating model at beta_min...')
params[param_name]=beta_min
T_min,logZ_min,obs_min,dNorm_min=eval_model(params)


print_and_log('evaluating model at beta_max...')
params[param_name]=beta_max
T_max,logZ_max,obs_max,dNorm_max=eval_model(params)


print_and_log('beta_min=',beta_min,'beta_max=',beta_max)
print_and_log('logZ_min=',logZ_min.item(),'logZ_max=',logZ_max.item())
print_and_log('obs_min=',obs_min.item(),'obs_max=',obs_max.item())
print_and_log('searching for critical temperature using bisection method')
beta_new=(beta_min+beta_max)/2
while beta_max-beta_min>options['tol']:
    beta_new=(beta_min+beta_max)/2
    if beta_new==beta_min or beta_new==beta_max:
        break
    params[param_name]=beta_new
    T_new,logZ_new,obs_new,dNorm_new=eval_model(params)
    print_and_log('beta_min=',beta_min,'beta_new=',beta_new,'beta_max=',beta_max,'beta_ref',beta_ref,'beta_diff=',beta_max-beta_min)
    print_and_log('logZ_min=',logZ_min.item(),'logZ_new=',logZ_new.item(),'logZ_max=',logZ_max.item())
    print_and_log('obs_min=',obs_min.item(),'obs_new=',obs_new.item(),'obs_max=',obs_max.item())
    if options['method']=='logZ':
        dist_min=(logZ_min-logZ_new).abs()
        dist_max=(logZ_max-logZ_new).abs()
    elif options['method']=='obs':
        dist_min=(obs_min-obs_new).abs()
        dist_max=(obs_max-obs_new).abs()
    elif options['method']=='all_norm':
        dist_min=(dNorm_min-dNorm_new).norm()
        dist_max=(dNorm_max-dNorm_new).norm()
    elif options['method']=='final_norm':
        dist_min=(dNorm_min[-1]-dNorm_new[-1]).abs()
        dist_max=(dNorm_max[-1]-dNorm_new[-1]).abs()
    #dist_min=contract('ijkl,ijkl->',T_min,T_new).abs()
    #dist_max=contract('ijkl,ijkl->',T_max,T_new).abs()
    print_and_log('dist_min=',dist_min,'dist_max=',dist_max)
    if dist_min<dist_max:
        print_and_log('keeping beta_max')
        beta_min=beta_new
        T_min=T_new
        logZ_min=logZ_new
        obs_min=obs_new
        dNorm_min=dNorm_new
    else:
        print_and_log('keeping beta_min')
        beta_max=beta_new
        T_max=T_new
        logZ_max=logZ_new
        obs_max=obs_new
        dNorm_max=dNorm_new


print_and_log('critical temperature found: beta=',beta_new,' reference: ',beta_ref)



filename=options['filename']

import os
dirname=os.path.dirname(filename)
if not os.path.exists(dirname):
    os.makedirs(dirname)


torch.save({param_name:beta_new},filename)

if logfile is not None:
    logfile.close()
    