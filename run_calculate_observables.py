if __name__ != '__main__':
    assert False, 'This file is not meant to be imported'

import argparse
parser = argparse.ArgumentParser()


parser.add_argument('--tensors_filename', type=str, required=True)
parser.add_argument('--output_filename', type=str, required=True)

parser.add_argument('--observables', type=str, nargs='+', required=True)
parser.add_argument('--calc_binder', action='store_true')
parser.add_argument('--double_layer', action='store_true')

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--override', action='store_true')
args = parser.parse_args()

options=vars(args)

tensors_filename=options['tensors_filename']
output_filename=options['output_filename']

import os
if os.path.exists(output_filename) and not options['override']:
    print('file already exists: ',output_filename)
    print('use --override to override')
    quit()


print('loading library...')
from opt_einsum import contract # idk why but its required to avoid bug in contract with numpy arrays
import torch
torch.set_default_tensor_type(torch.DoubleTensor if options['device']=='cpu' else torch.cuda.DoubleTensor)
device=torch.device(options['device']);torch.cuda.set_device(device)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from HOTRG import forward_observalbe_tensor_moments,trace_tensor,trace_two_tensors
import TNModels

print('calculating observables...')
# load the tensor file
print('loading',tensors_filename)
tensor_data=torch.load(tensors_filename,map_location=device)
options={**tensor_data['options'],**options}
params,layers,Ts,logTotals=tensor_data['params'],tensor_data['layers'],tensor_data['Ts'],tensor_data['logTotals']

Model=TNModels.Models[options['model']]
model=Model(params)

data={}
for observable_name in options['observables']:
    n_moments=4 if options['calc_binder'] else 1
    print('calculating',observable_name)
    T_op0_moments,checkerboard=model.get_observable_moments(observable_name,n=n_moments)
    T_op_momentss,logTotals=forward_observalbe_tensor_moments(T_op0_moments,layers,checkerboard=checkerboard,return_layers=True,cached_Ts=Ts)

    for iLayer in tqdm(range(len(Ts)),leave=False):
        logTotal=logTotals[iLayer]
        T=T_op_momentss[iLayer][0]
        T_op=T_op_momentss[iLayer][1]

        logZ=(logTotal+trace_tensor(T).abs().log())/2**iLayer
        moment1=(trace_two_tensors(T_op)/trace_two_tensors(T)).abs().sqrt()

        key=(iLayer,tuple(params.values()))

        data[key]={
            **data.get(key,{}),
            **params,
            'max_dim':options['max_dim'],
            'iLayer':iLayer,
            'logZ':logZ.item(),
            observable_name:moment1.item(),
        }
        if options['calc_binder']:
            T_op2=T_op_momentss[iLayer][2]
            T_op3=T_op_momentss[iLayer][3]
            T_op4=T_op_momentss[iLayer][4]
            moment2=trace_tensor(T_op2)/trace_tensor(T)
            moment4=trace_tensor(T_op4)/trace_tensor(T)
            data[key].update({
                observable_name+'_2':moment2.item(),
                observable_name+'_4':moment4.item(),
            })
        if options['double_layer']:
            obs_double=(trace_two_tensors(T_op,T_op)/trace_two_tensors(T,T)).abs().sqrt()
            data[key].update({
                observable_name+'_double':obs_double.item(),
            })


os.makedirs(os.path.dirname(output_filename),exist_ok=True)
df=pd.DataFrame(data.values())
df.to_csv(output_filename,index=False)
print('saved to',output_filename)