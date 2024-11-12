from .math_module import xp, xcipy, ensure_np_array
from adefc_vortex import utils
from adefc_vortex.imshows import imshow1, imshow2, imshow3

import numpy as np
import scipy
import astropy.units as u
from astropy.io import fits
import poppy
import time
import copy
import os
from pathlib import Path
from IPython.display import clear_output
from datetime import datetime
today = int(datetime.today().strftime('%Y%m%d'))

try:
    from scoobpy import utils as scoob_utils
    import purepyindi
    import purepyindi2
    from magpyx.utils import ImageStream
    import ImageStreamIOWrap as shmio
except ImportError:
    print('SCoOB interface does not have the required packages to operate.')

def create_shmim(name, dims, dtype=shmio.ImageStreamIODataType.FLOAT, shared=1, nbkw=8):
    # if ImageStream objects didn't auto-open on creation, you could create and return that instead. oops.
    img = shmio.Image()
    # not sure if I should try to destroy first in case it already exists
    # img.create(name, dims, dtype, shared, nbkw)
    buffer = np.zeros(dims)
    img.create(name, buffer, -1, True, 8, 1, dtype, 1)

def move_psf(x_pos, y_pos, client):
    client.wait_for_properties(['stagepiezo.stagepupil_x_pos', 'stagepiezo.stagepupil_y_pos'])
    scoob_utils.move_relative(client, 'stagepiezo.stagepupil_x_pos', x_pos)
    time.sleep(0.25)
    scoob_utils.move_relative(client, 'stagepiezo.stagepupil_y_pos', y_pos)
    time.sleep(0.25)

def home_block(client, delay=2):
    client.wait_for_properties(['stagelinear.home'])
    client['stagelinear.home.request'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def move_block_in(client, delay=2):
    client.wait_for_properties(['stagelinear.presetName'])
    client['stagelinear.presetName.block_in'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def move_block_out(client, delay=2):
    client.wait_for_properties(['stagelinear.presetName'])
    client['stagelinear.presetName.block_out'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def set_zwo_roi(xc, yc, npix, client, delay=0.25):
    # update roi parameters
    client.wait_for_properties(['camsci.roi_region_x', 'camsci.roi_region_y', 'camsci.roi_region_h' ,'camsci.roi_region_w', 'camsci.roi_set'])
    client['camsci.roi_region_x.target'] = xc
    client['camsci.roi_region_y.target'] = yc
    client['camsci.roi_region_h.target'] = npix
    client['camsci.roi_region_w.target'] = npix
    time.sleep(delay)
    client['camsci.roi_set.request'] = purepyindi.SwitchState.ON
    time.sleep(delay)

class SCOOBI():

    def __init__(self, 
                 dm_channel,
                 scicam_channel=None,
                 locam_channel=None,
                 dm_ref=np.zeros((34,34)),
                 npsf=150,
                ):
        self.wavelength_c = 633e-9*u.m
        
        self.SCICAM = ImageStream(scicam_channel) if scicam_channel is not None else None
        self.LOCAM = ImageStream(locam_channel) if locam_channel is not None else None
        self.DM = scoob_utils.connect_to_dmshmim(channel=dm_channel) # channel used for writing to DM
        # self.DM_WFE = scoob_utils.connect_to_dmshmim(channel=wfe_channel) if wfe_channel is not None else None
        self.DMT = scoob_utils.connect_to_dmshmim(channel='dm00disp') # the total shared memory image
        self.dm_delay = 0.1

        # Init all DM settings
        self.Nact = 34
        self.Nacts = 951 # accounting for the bad actuator
        self.dm_shape = (self.Nact,self.Nact)
        self.act_spacing = 300e-6*u.m
        self.dm_active_diam = 10.2*u.mm
        self.dm_full_diam = 11.1*u.mm
        self.full_stroke = 1.5e-6*u.m
        self.dm_ref = dm_ref
        self.dm_gain = 1
        self.reset_dm()
        
        self.bad_acts = [(25,21)]
        xx = (np.linspace(0, self.Nact-1, self.Nact) - self.Nact/2 + 1/2) * self.act_spacing.to_value(u.mm)
        x,y = np.meshgrid(xx,xx)
        r = np.sqrt(x**2 + y**2)
        self.dm_mask = r<10.5/2
        self.dm_pupil_mask = r<9.1/2

        # Init camera settings
        self.psf_pixelscale = 4.6e-6*u.m/u.pix
        self.psf_pixelscale_lamD = 0.307
        self.nbits = 16
        self.NSCICAM = 1
        self.NLOCAM = 1
        self.npsf = npsf
        self.nlocam = 100
        self.x_shift = 0
        self.y_shift = 0
        self.x_shift_locam = 0
        self.y_shift_locam = 0

        self.atten = 1
        self.texp = 1
        self.gain = 1
        self.texp_locam = 1
        self.gain_locam = 1
        
        self.ref_psf_params = None
        self.dark_frame = None

        self.subtract_dark = False
        self.return_ni = False

    def getattr(self, attr):
        return getattr(self, attr)
    
    def setattr(self, attr, val):
        setattr(self, attr, val)

    def set_fib_atten(self, value, client, delay=0.1):
        client['fiberatten.atten.target'] = value
        time.sleep(delay)
        self.atten = value
        print(f'Set the fiber attenuation to {value:.1f}')

    def set_nsv_exp_time(self, exp_time, client, delay=0.25):
        if exp_time<1e-4:
            print('Minimum exposure time is 1E-4 seconds. Setting exposure time to minimum.')
            exp_time = 1e-4
        client.wait_for_properties(['nsv571.exptime'])
        client['nsv571.exptime.target'] = exp_time
        time.sleep(delay)
        self.texp_locam = exp_time
        print(f'Set the NSV571 exposure time to {self.texp_locam:.2e}s')

    def set_nsv_gain(self, gain, client, delay=0.25):
        client.wait_for_properties(['nsv571.emgain'])
        client['nsv571.emgain.target'] = gain
        time.sleep(delay)
        self.gain_locam = gain
        print(f'Set the NSV571 gain to {self.texp_locam:.2e}s')
        
    def set_zwo_exp_time(self, exp_time, client, delay=0.25):
        if exp_time<3.2e-5:
            print('Minimum exposure time is 3.2E-5 seconds. Setting exposure time to minimum.')
            exp_time = 3.2e-5
        client.wait_for_properties(['camsci.exptime'])
        client['camsci.exptime.target'] = exp_time
        time.sleep(delay)
        self.texp = exp_time
        print(f'Set the ZWO exposure time to {self.texp:.2e}s')

    def set_zwo_gain(self, gain, client, delay=0.1):
        client.wait_for_properties(['camsci.emgain'])
        client['camsci.emgain.target'] = gain
        time.sleep(delay)
        self.gain = gain
        print(f'Set the ZWO gain setting to {gain:.1f}')
    
    def zero_dm(self):
        self.DM.write(np.zeros(self.dm_shape))
        time.sleep(self.dm_delay)
    
    def reset_dm(self):
        self.DM.write(ensure_np_array(self.dm_ref))
        time.sleep(self.dm_delay)
    
    def set_dm(self, dm_command):
        self.DM.write(ensure_np_array(dm_command)*1e6)
        time.sleep(self.dm_delay)
    
    def add_dm(self, dm_command):
        dm_state = ensure_np_array(self.get_dm())
        self.DM.write( 1e6*(dm_state + ensure_np_array(dm_command)) )
        time.sleep(self.dm_delay)
               
    def get_dm(self):
        return xp.array(self.DM.grab_latest())/1e6
    
    def close_dm(self):
        self.DM.close()

    def normalize(self, image):
        if self.ref_psf_params is None:
            raise ValueError('Cannot normalize because reference PSF not specified.')
        image_ni = image/self.ref_psf_params['Imax']
        image_ni *= (self.ref_psf_params['texp']/self.texp)
        image_ni *= 10**((self.atten-self.ref_psf_params['atten'])/10)
        image_ni *= 10**(-self.gain/20 * 0.1) / 10**(-self.ref_psf_params['gain']/20 * 0.1)
        return image_ni

    def snap(self, normalize=False, plot=False, vmin=None):
        if self.NSCICAM>1:
            ims = self.SCICAM.grab_many(self.NSCICAM)
            im = np.sum(ims, axis=0)/self.NSCICAM
        else:
            im = self.SCICAM.grab_latest()
        
        im = xp.array(im)
        im = xcipy.ndimage.shift(im, (self.y_shift, self.x_shift), order=0)
        im = utils.pad_or_crop(im, self.npsf)

        if self.subtract_dark and self.df is not None:
            im -= self.df
            im[im<0] = 0.0
            
        if self.return_ni:
            im = self.normalize(im)
        
        return im
    
def stream_scicam(I, duration=60, control_mask=None, plot=False, clear=True, save_data_to=None):
    I.subtract_dark = True
    I.return_ni = True

    all_ims = []
    try:
        print('Streaming camsci data ...')
        i = 0
        start = time.time()
        while (time.time()-start)<duration:
            im = I.snap()
            i += 1
            if save_data_to is not None:
                all_ims.append(im)
            if control_mask is not None:
                mean_ni = xp.mean(im[control_mask])
                print(f'Mean NI = {mean_ni:.2e}')
            if plot:
                imshows.imshow1(im, lognorm=True, vmin=1e-9)
            if clear:
                clear_output(wait=True)
    except KeyboardInterrupt:
        print('Stopping camsci stream!')
    if save_data_to is not None:
        scoobi.utils.save_fits(save_data_to, xp.array(all_ims))
        
        
