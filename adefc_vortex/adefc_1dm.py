from .math_module import xp, xcipy, ensure_np_array
from adefc_vortex import utils
from adefc_vortex.imshows import imshow1, imshow2, imshow3

import numpy as np
from scipy.optimize import minimize
import time
import copy

def run_pwp(I, 
            M, 
            control_mask, 
            probes, probe_amp, 
            reg_cond=1e-3, 
            plot=False,
            plot_est=False,
            ):
    
    Nmask = int(control_mask.sum())
    Nprobes = probes.shape[0]

    current_acts = I.get_dm()[M.dm_mask]

    I.subtract_dark = False
    Ip = []
    In = []
    for i in range(Nprobes):
        for s in [-1, 1]:
            I.add_dm(s*probe_amp*probes[i])
            coro_im = I.snap()
            I.add_dm(-s*probe_amp*probes[i]) # remove probe from DM

            if s==-1: 
                In.append(coro_im)
            else: 
                Ip.append(coro_im)
        
    E_probes = xp.zeros((probes.shape[0], 2*Nmask))
    I_diff = xp.zeros((probes.shape[0], Nmask))
    for i in range(Nprobes):
        if i==0: 
            E_nom = M.forward(current_acts, use_vortex=True)
        E_with_probe = M.forward(xp.array(current_acts) + xp.array(probe_amp*probes[i])[M.dm_mask], use_vortex=True)
        E_probe = E_with_probe - E_nom
        diff_im = Ip[i] - In[i]
        if plot:
            imshow3(diff_im, xp.abs(E_probe), xp.angle(E_probe),
                    'Difference Image', f'Probe {i+1}: '+'$|E_{probe}|$', f'Probe {i+1}: '+r'$\angle E_{probe}$', 
                    cmap3='twilight')
            
        E_probes[i, ::2] = E_probe[control_mask].real
        E_probes[i, 1::2] = E_probe[control_mask].imag
        I_diff[i, :] = diff_im[control_mask]
    
    # Use batch process to estimate each pixel individually
    E_est = xp.zeros(Nmask, dtype=xp.complex128)
    for i in range(Nmask):
        delI = I_diff[:, i]
        H = 4*xp.array([E_probes[:,2*i], E_probes[:,2*i + 1]]).T
        Hinv = xp.linalg.pinv(H.T@H, reg_cond)@H.T
    
        est = Hinv.dot(delI)

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = xp.zeros((I.npsf,I.npsf), dtype=xp.complex128)
    E_est_2d[control_mask] = E_est

    if plot or plot_est:
        I_est = xp.abs(E_est_2d)**2
        P_est = xp.angle(E_est_2d)
        imshow2(I_est, P_est, 
                'Estimated Intensity', 'Estimated Phase',
                lognorm1=True, vmin1=xp.max(I_est)/1e3, 
                cmap2='twilight',
                pxscl=M.psf_pixelscale_lamD)
        
    return E_est_2d

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
        print('Running estimation algorithm ...')
        
        if pwp_params is not None: 
            E_ab = run_pwp(I, M, **pwp_params)
        else:
            E_ab = I.calc_wf()
        
        print('Computing EFC command with L-BFGS')
        current_acts = total_command[M.dm_mask]
        E_FP_NOM, E_EP, DM_PHASOR = M.forward(current_acts, I.wavelength_c, use_vortex=True, return_ints=True)
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


