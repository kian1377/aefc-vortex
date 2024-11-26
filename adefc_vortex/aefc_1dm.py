from .math_module import xp, xcipy, ensure_np_array
from adefc_vortex import utils
from adefc_vortex.imshows import imshow1, imshow2, imshow3
import adefc_vortex.pwp as pwp

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

        if pwp_params is not None: 
            print('Running PWP ...')
            E_ab = pwp.run(I, M, **pwp_params)
        else:
            print('Computing E-field with model ...')
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

import matplotlib.pyplot as plt
plt.rcParams['image.origin']='lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec

def plot_pwp(probes, E_probes, diff_ims, E_est, vmin=1e-8, vmax=1e-4):
    probes = ensure_np_array(probes)
    E_probes = ensure_np_array(E_probes)
    diff_ims = ensure_np_array(diff_ims)
    E_est = ensure_np_array(E_est)

    fig = plt.figure(figsize=(20, 15), dpi=125)
    gs = GridSpec(3, 4, figure=fig)

    title_fz = 16

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(probes[0], cmap='viridis',)
    ax.set_title('Probe 1', fontsize=title_fz)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(np.abs(E_probes[0]), cmap='magma',)
    ax.set_title('Probe 1 Model-based Amplitude', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(np.angle(E_probes[0]), cmap='twilight',)
    ax.set_title('Probe 1 Model-based Phase', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(diff_ims[0], cmap='magma',)
    ax.set_title('Probe 1 Difference Image', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(probes[1], cmap='viridis',)
    ax.set_title('Probe 2', fontsize=title_fz)

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(np.abs(E_probes[1]), cmap='magma',)
    ax.set_title('Probe 2 Model-based Amplitude', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(np.angle(E_probes[1]), cmap='twilight',)
    ax.set_title('Probe 2 Model-based Phase', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(diff_ims[1], cmap='magma',)
    ax.set_title('Probe 2 Difference Image', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[2, :2])
    im = ax.imshow(np.abs(E_est)**2, cmap='magma', norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_title('Final Estimated Intensity: ' + r'$|E_{ab}|^2$', fontsize=title_fz+4)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im, cax=cax,)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_ylabel('NI', rotation=0, labelpad=10, fontsize=14)
    ax.set_position([0.2, 0.025, 0.3, 0.3]) # [left, bottom, width, height]

    ax = fig.add_subplot(gs[2, 2:])
    im = ax.imshow(np.angle(E_est), cmap='twilight',)
    ax.set_title('Final Estimated Phase: ' + r'$\angle E_{ab}$', fontsize=title_fz+4)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=14)
    ax.set_position([0.55, 0.025, 0.3, 0.3]) # [left, bottom, width, height]


