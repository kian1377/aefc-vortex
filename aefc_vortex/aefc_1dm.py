from .math_module import xp, xcipy, ensure_np_array
from aefc_vortex import utils
from aefc_vortex.imshows import imshow1, imshow2, imshow3
import aefc_vortex.pwp as pwp

import numpy as np
from scipy.optimize import minimize
import time
import copy

def run(I, 
        M, 
        val_and_grad,
        control_mask,
        data,
        pwp_params=None,
        Nitr=3, 
        reg_cond=1e-2,
        bfgs_tol=1e-3,
        bfgs_opts=None,
        gain=0.5, 
        leakage=0.0, 
        vmin=1e-9, 
        ):

    starting_itr = len(data['images'])

    total_command = copy.copy(data['commands'][-1]) if len(data['commands'])>0 else xp.zeros((M.Nact,M.Nact))

    del_command = xp.zeros((M.Nact,M.Nact)) # array to fill with actuator solutions
    del_acts0 = np.zeros(M.Nacts) # initial guess is always just zeros
    for i in range(Nitr):
        print(f'Running iteration {starting_itr+i:d}')

        if pwp_params is not None: 
            print('Running PWP ...')
            E_ab = pwp.run(I, M, **pwp_params)
        else:
            print('Computing E-field with model ...')
            E_ab = I.calc_wf()
        
        current_acts = total_command[M.dm_mask]
        E_FP_NOM, E_EP, DM_PHASOR, _, _, _ = M.forward(current_acts, I.wavelength_c, use_vortex=True, return_ints=True)
        rmad_vars= {
            'E_ab': E_ab,
            'current_acts': current_acts,
            'E_FP_NOM': E_FP_NOM, 
            'E_EP': E_EP, 
            'DM_PHASOR': DM_PHASOR,
            'control_mask': control_mask,
            'wavelength':I.wavelength_c,
            'r_cond': reg_cond, 
        }

        res = minimize(
            val_and_grad, 
            jac=True, 
            x0=del_acts0,
            args=(M, rmad_vars, 0, 0, 0), 
            method='L-BFGS-B',
            tol=bfgs_tol,
            options=bfgs_opts,
        )

        del_acts = gain * res.x
        del_command[M.dm_mask] = del_acts
        total_command = (1-leakage)*total_command + del_command
        I.set_dm(total_command)

        I.return_ni = True
        I.subtract_dark = True
        image_ni = I.snap()
        mean_ni = xp.mean(image_ni[control_mask])

        data['images'].append(copy.copy(image_ni))
        data['efields'].append(copy.copy(E_ab))
        data['commands'].append(copy.copy(total_command))
        data['del_commands'].append(copy.copy(del_command))
        data['bfgs_tols'].append(bfgs_tol)
        data['reg_conds'].append(reg_cond)
        
        imshow3(del_command, total_command, image_ni, 
                f'Iteration {starting_itr + i:d}: $\delta$DM', 
                'Total DM Command', 
                f'Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                vmin1=-xp.max(xp.abs(del_command)), vmax1=xp.max(xp.abs(del_command)),
                vmin2=-xp.max(xp.abs(total_command)), vmax2=xp.max(xp.abs(total_command)),
                pxscl3=I.psf_pixelscale_lamDc, lognorm3=True, vmin3=vmin)

    return data

