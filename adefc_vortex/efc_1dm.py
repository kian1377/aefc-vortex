from .math_module import xp, xcipy, ensure_np_array
from adefc_vortex import utils
from adefc_vortex.imshows import imshow1, imshow2, imshow3
import adefc_vortex.pwp as pwp

import numpy as np
from scipy.optimize import minimize
import time
import copy

def compute_jacobian(M, control_mask, amp=1e-9, current_acts=None, wavelength=None):
    if current_acts is None:
        current_acts = xp.zeros(M.Nacts)
    if wavelength is None:
        wavelength = M.wavelength_c

    Nmask = int(control_mask.sum())
    jac = xp.zeros((2*Nmask, M.Nacts))
    print(jac.shape)
    start = time.time()
    for i in range(M.Nacts):
        del_acts = xp.zeros(M.Nacts)
        del_acts[i] = amp
        E_pos = M.forward(current_acts + del_acts, wavelength, use_vortex=1, )
        E_neg = M.forward(current_acts - del_acts, wavelength, use_vortex=1, )
        response = ( E_pos - E_neg ) / (2*amp)
        jac[::2, i] = response.real[control_mask]
        jac[1::2, i] = response.imag[control_mask]
        print(f"\tCalibrated mode {i+1:d}/{M.Nacts:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")

    return jac

def run(I, 
        control_matrix,
        control_mask,
        data,
        pwp_params=None,
        Nitr=3, 
        gain=0.5, 
        ):
    
    starting_itr = len(data['images'])
    total_command = copy.copy(data['commands'][-1]) if len(data['commands'])>0 else xp.zeros((I.Nact,I.Nact))

    del_command = xp.zeros((I.Nact,I.Nact)) # array to fill with actuator solutions
    Nacts = control_matrix.shape[0]
    Nmask = int(control_mask.sum())
    E_ab_vec = xp.zeros(2*Nmask)
    for i in range(Nitr):
        
        if pwp_params is not None: 
            print('Running PWP ...')
            E_ab = pwp.run(I, **pwp_params)
        else:
            print('Computing E-field with model ...')
            E_ab = I.calc_wf()

        E_ab_vec[::2] = E_ab[control_mask].real
        E_ab_vec[1::2] = E_ab[control_mask].imag
        del_acts = - gain * control_matrix.dot(E_ab_vec)
        del_command[I.dm_mask] = del_acts[:Nacts]
        total_command += del_command
        I.set_dm(total_command)

        I.return_ni = True
        I.subtract_dark = True
        image_ni = I.snap()
        mean_ni = xp.mean(image_ni[control_mask])

        data['images'].append(copy.copy(image_ni))
        data['efields'].append(copy.copy(E_ab))
        data['commands'].append(copy.copy(total_command))
        data['del_commands'].append(copy.copy(del_command))

        imshow3(del_command, total_command, image_ni, 
                f'$\delta$DM1', f'$\delta$DM2', 
                f'Iteration {starting_itr + i:d} Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                pxscl3=I.psf_pixelscale_lamDc, lognorm3=True, vmin3=1e-10)

    return data
