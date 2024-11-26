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

def compute_jacobian_bb(M, control_mask, waves, amp=1e-9, current_acts=None,):

    Nwaves = waves.shape[0]
    Nmask = int(control_mask.sum())
    jac = xp.zeros((Nwaves * 2*Nmask, M.Nacts))
    print(jac.shape)
    for i in range(Nwaves):
        mono_jac = compute_jacobian(M, control_mask, amp, current_acts, waves[i])
        jac[i*2*Nmask:(i+1)*2*Nmask] = mono_jac

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
    if len(data['dm1_commands'])>0:
        total_dm1 = copy.copy(data['dm1_commands'][-1])
        total_dm2 = copy.copy(data['dm2_commands'][-1])
    else:
        total_dm1, total_dm2 = ( xp.zeros((I.Nact,I.Nact)), xp.zeros((I.Nact,I.Nact)) ) 

    del_dm1 = xp.zeros((I.Nact,I.Nact))
    del_dm2 = xp.zeros((I.Nact,I.Nact))
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
        del_dm1[I.dm_mask] = del_acts[:Nacts//2]
        del_dm2[I.dm_mask] = del_acts[Nacts//2:]
        total_dm1 += del_dm1
        total_dm2 += del_dm2

        I.add_dm1(del_dm1)
        I.add_dm2(del_dm2)
        image_ni = I.snap()
        mean_ni = xp.mean(image_ni[control_mask])

        data['images'].append(copy.copy(image_ni))
        data['efields'].append(copy.copy(E_ab))
        data['dm1_commands'].append(copy.copy(total_dm1))
        data['del_dm1_commands'].append(copy.copy(del_dm1))
        data['dm2_commands'].append(copy.copy(total_dm2))
        data['del_dm2_commands'].append(copy.copy(del_dm2))
        
        imshow3(del_dm1, del_dm2, image_ni, 
                f'$\delta$DM1', f'$\delta$DM2', 
                f'Iteration {starting_itr + i:d} Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                pxscl3=I.psf_pixelscale_lamDc, lognorm3=True, vmin3=1e-10)

    return data

def calc_wfs(I, waves, control_mask, plot=False):
    Nwaves = len(waves)
    E_abs = xp.zeros((Nwaves, I.npsf, I.npsf), dtype=xp.complex128)
    for i in range(Nwaves):
        I.wavelength = waves[i]
        E_abs[i] = I.calc_wf() * control_mask
        if plot: imshow2(xp.abs(E_abs[i])**2, xp.angle(E_abs[i])*control_mask, lognorm1=True, cmap2='twilight')

    return E_abs

def run_bb(I, 
        control_matrix,
        control_mask,
        control_waves,
        data,
        pwp_params=None,
        Nitr=3, 
        gain=0.5, 
        ):
    
    starting_itr = len(data['images'])
    if len(data['dm1_commands'])>0:
        total_dm1 = copy.copy(data['dm1_commands'][-1])
        total_dm2 = copy.copy(data['dm2_commands'][-1])
    else:
        total_dm1, total_dm2 = ( xp.zeros((I.Nact,I.Nact)), xp.zeros((I.Nact,I.Nact)) ) 

    del_dm1 = xp.zeros((I.Nact,I.Nact))
    del_dm2 = xp.zeros((I.Nact,I.Nact))
    Nacts = control_matrix.shape[0]
    Nmask = int(control_mask.sum())
    Nwaves = control_waves.shape[0]
    E_ab_vec = xp.zeros(Nwaves * 2*Nmask)
    for i in range(Nitr):
        if pwp_params is not None: 
            print('Running PWP ...')
            E_abs = pwp.run_bb(I, M, **pwp_params)
        else:
            print('Computing E-field with model ...')
            E_abs = calc_wfs(I, control_waves, control_mask)

        for i in range(Nwaves):
            E_ab_vec[i*2*Nmask:(i+1)*2*Nmask][::2] = E_abs[i][control_mask].real
            E_ab_vec[i*2*Nmask:(i+1)*2*Nmask][1::2] = E_abs[i][control_mask].imag

        del_acts = - gain * control_matrix.dot(E_ab_vec)
        del_dm1[I.dm_mask] = del_acts[:Nacts//2]
        del_dm2[I.dm_mask] = del_acts[Nacts//2:]
        total_dm1 += del_dm1
        total_dm2 += del_dm2

        I.add_dm1(del_dm1)
        I.add_dm2(del_dm2)
        image_ni = I.snap()
        mean_ni = xp.mean(image_ni[control_mask])

        data['images'].append(copy.copy(image_ni))
        data['efields'].append(copy.copy(E_abs))
        data['dm1_commands'].append(copy.copy(total_dm1))
        data['del_dm1_commands'].append(copy.copy(del_dm1))
        data['dm2_commands'].append(copy.copy(total_dm2))
        data['del_dm2_commands'].append(copy.copy(del_dm2))
        
        imshow3(del_dm1, del_dm2, image_ni, 
                f'$\delta$DM1', f'$\delta$DM2', 
                f'Iteration {starting_itr + i:d} Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                pxscl3=I.psf_pixelscale_lamDc, lognorm3=True, vmin3=1e-10)

    return data

# def run_bb(
#         I, 
#         M, 
#         val_and_grad,
#         control_mask,
#         data,
#         pwp_params=None,
#         Nitr=3, 
#         reg_cond=1e-2,
#         weights=None, 
#         bfgs_tol=1e-3,
#         bfgs_opts=None,
#         gain=0.5, 
#     ):
    
#     Nbps = I.bandpasses.shape[0]
#     Nwaves_per_bp = I.bandpasses.shape[1]
#     control_waves = I.bandpasses[:, Nwaves_per_bp//2]

#     starting_itr = len(data['images'])
#     if len(data['dm1_commands'])>0:
#         total_dm1, total_dm2 = ( copy.copy(data['dm1_commands'][-1]), copy.copy(data['dm2_commands'][-1]) )
#     else:
#         total_dm1, total_dm2 = ( xp.zeros((M.Nact,M.Nact)), xp.zeros((M.Nact,M.Nact)) ) 

#     del_dm1 = xp.zeros((M.Nact,M.Nact))
#     del_dm2 = xp.zeros((M.Nact,M.Nact))
#     del_acts0 = np.zeros(M.Nacts)
#     for i in range(Nitr):
#         if pwp_params is not None: 
#             print('Running PWP ...')
#             E_abs = pwp.run_bb(I, M, **pwp_params)
#         else:
#             print('Computing E-field with model ...')
#             E_abs = calc_wfs(I, control_waves, control_mask)
        
#         print('Computing EFC command with L-BFGS')
#         current_acts = xp.concatenate([total_dm1[M.dm_mask], total_dm2[M.dm_mask]])
#         E_FP_NOMs, E_EPs, E_DM2Ps, DM1_PHASORs, DM2_PHASORs = get_forward_vars(M, current_acts, control_waves)
#         rmad_vars = { 
#             'current_acts':current_acts,
#             'E_abs':E_abs, 
#             'E_FP_NOMs':E_FP_NOMs,
#             'E_EPs':E_EPs,
#             'E_DM2Ps':E_DM2Ps,
#             'DM1_PHASORs':DM1_PHASORs,
#             'DM2_PHASORs':DM2_PHASORs,
#             'control_mask':control_mask,
#             'control_waves':control_waves,
#             'r_cond':reg_cond,
#             'weights':weights,
#         }

#         res = minimize(
#             val_and_grad, 
#             jac=True, 
#             x0=del_acts0,
#             args=(M, rmad_vars), 
#             method='L-BFGS-B',
#             tol=bfgs_tol,
#             options=bfgs_opts,
#         )

#         del_acts = gain * res.x
#         del_dm1[M.dm_mask] = del_acts[:M.Nacts//2]
#         del_dm2[M.dm_mask] = del_acts[M.Nacts//2:]
#         I.add_dm1(del_dm1)
#         I.add_dm2(del_dm2)
#         total_dm1, total_dm2 = ( total_dm1 + del_dm1, total_dm2 + del_dm2 )

#         image_ni = I.snap()
#         mean_ni = xp.mean(image_ni[control_mask])

#         data['images'].append(copy.copy(image_ni))
#         data['efields'].append(copy.copy(E_abs))
#         data['dm1_commands'].append(copy.copy(total_dm1))
#         data['del_dm1_commands'].append(copy.copy(del_dm1))
#         data['dm2_commands'].append(copy.copy(total_dm2))
#         data['del_dm2_commands'].append(copy.copy(del_dm2))
#         data['bfgs_tols'].append(bfgs_tol)
#         data['reg_conds'].append(reg_cond)
        
#         imshow3(del_dm1, del_dm2, image_ni, 
#                 f'$\delta$DM1', f'$\delta$DM2', 
#                 f'Iteration {starting_itr + i:d} Image\nMean NI = {mean_ni:.3e}',
#                 cmap1='viridis', cmap2='viridis', 
#                 pxscl3=I.psf_pixelscale_lamDc, lognorm3=True, vmin3=1e-10)

#     return data


