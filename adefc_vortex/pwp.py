from .math_module import xp, xcipy, ensure_np_array
from adefc_vortex import utils
from adefc_vortex.imshows import imshow1, imshow2, imshow3

import numpy as np
import time
import copy

import matplotlib.pyplot as plt
plt.rcParams['image.origin']='lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec

def run(I, 
        M, 
        control_mask, 
        probes, probe_amp, 
        wavelength, 
        reg_cond=1e-3, 
        gain=1,
        Ndms=1,
        plot=False,
        plot_est=False,
        plot_fname=None, 
        return_all=False,
        ):
    
    Nmask = int(control_mask.sum())
    Nprobes = probes.shape[0]

    if Ndms==1:
        current_acts = I.get_dm()[M.dm_mask]
    elif Ndms==2: 
        current_acts = xp.concatenate([I.get_dm1()[M.dm_mask], I.get_dm2()[M.dm_mask]])

    I.subtract_dark = False
    Ip = []
    In = []
    for i in range(Nprobes):
        for s in [-1, 1]:
            if Ndms==1:
                I.add_dm(s*probe_amp*probes[i])
                coro_im = I.snap()
                I.add_dm(-s*probe_amp*probes[i]) # remove probe from DM
            elif Ndms==2: 
                I.add_dm1(s*probe_amp*probes[i])
                coro_im = I.snap()
                I.add_dm1(-s*probe_amp*probes[i]) # remove probe from DM

            if s==-1: 
                In.append(coro_im)
            else: 
                Ip.append(coro_im)
    
    E_probes = xp.zeros((probes.shape[0], I.npsf, I.npsf), dtype=xp.complex128)
    diff_ims = xp.zeros((probes.shape[0], I.npsf, I.npsf))
    for i in range(Nprobes):
        if i==0: 
            E_nom = M.forward(current_acts, wavelength, use_vortex=True)
        if Ndms==1:
            probe_acts = xp.array(probe_amp*probes[i])[M.dm_mask]
        else: 
            probe_acts = xp.concatenate([probe_amp*probes[i][M.dm_mask], xp.zeros(M.Nacts//2)])
        E_with_probe = M.forward(current_acts + probe_acts, wavelength, use_vortex=True)

        E_probes[i] = E_with_probe - E_nom
        diff_ims[i] = Ip[i] - In[i]
    
    # Use batch process to estimate each pixel individually
    E_est = xp.zeros(Nmask, dtype=xp.complex128)
    for i in range(Nmask):
        delI = diff_ims[:, control_mask][:, i]
        H = 4*xp.array(
            [E_probes[:, control_mask][:, i].real, 
             E_probes[:, control_mask][:, i].imag]
        ).T # Dimensions are 2 X N_probes
        Hinv = xp.linalg.pinv(H.T@H, reg_cond)@H.T
    
        est = Hinv.dot(delI)

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = xp.zeros((I.npsf, I.npsf), dtype=xp.complex128)
    E_est_2d[control_mask] = gain * E_est

    if plot:
        I_est = ensure_np_array( xp.abs(E_est_2d)**2 )
        plot_pwp(probes, E_probes, diff_ims, E_est_2d, vmin=np.max(I_est)/1e4, vmax=np.max(I_est), fname=plot_fname)
    if plot_est:
        I_est = xp.abs(E_est_2d)**2
        P_est = xp.angle(E_est_2d)
        imshow2(I_est, P_est, 
                'Estimated Intensity', 'Estimated Phase',
                lognorm1=True, vmin1=xp.max(I_est)/1e4, 
                cmap2='twilight',
                pxscl=I.psf_pixelscale_lamDc)

    if return_all:
        return E_est_2d, E_probes, diff_ims
    else:
        return E_est_2d

def plot_pwp(probes, E_probes, diff_ims, E_est, vmin=1e-9, vmax=1e-4, fname=None):
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

    plt.show()

    if fname is not None: fig.savefig(fname, format='pdf', bbox_inches="tight")




