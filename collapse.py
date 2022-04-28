import cryptography
import pandas as pd
import glob
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
import attr
import pdb
import snr
import pyklip.klip as klip
import pyklip.instruments.MagAO as MagAO
import pyklip.parallelized as parallelized
import pyklip.fakes as fakes
    

@attr.s(auto_attribs=True)
class PECollapser:

    pepath: str
    """Full path of parameter explorer output fits file"""

    handle_planets: str
    """mean or median across planet dimensions (e.g. "mean")"""

    dataname: str
    """Name of the dataset to be stored e.g. "HD142527_15May15" """

    ra: float
    """Radius or separation of planet in pixels e.g. [24.5]"""

    pa: float
    """Position angle of planet in degrees e.g. [170]"""

    hp: float
    """Highpass filter size for KLIP reductions in pixels (usually set to 1/2 FWHM)"""

    iwa: float
    """ IWA for KLIP reductions (usually set to FWHM)"""

    def collapse_across_planets(self) -> dict:
        """Collapse parameter explorer across the planet dimension 

        The output will identify the best parameter combinations for each metric

        Returns:
            dict: PeakSNR_annuli, PeakSNR_movement, PeakSNR_kl, AvgSNR_annuli, AvgSNR_movement, AvgSNR_kl, Contrast_annuli, Contrast_movement, Contrast_kl
        """

        # Load fits file
        data = fits.getdata(self.pepath)

        # Collapse across planet dimesnion (reduce to only kl, movement, annuli dimensions)
        if self.handle_planets == 'mean':
            collapsed = np.nanmean(data, axis = 3)
        elif self.handle_planets == 'median':
            collapsed = np.nanmedian(data, axis = 3)

        # Find best parameters across all metrics
        self.peaksnr_ann, self.peaksnr_move, self.peaksnr_kl = self.collapse_peaksnr(collapsed)
        self.avgsnr_ann, self.avgsnr_move, self.avgsnr_kl = self.collapse_avgsnr(collapsed)
        self.contrast_ann, self.contrast_move, self.contrast_kl = self.collapse_contrast(collapsed)
        self.all_ann, self.all_move, self.all_kl = self.collapse_all(collapsed)

        results = {"Dataname": self.dataname,
                    "RA": self.ra,
                    "PA": self.pa,
                    "highpass": self.hp,
                    "IWA": self.iwa,
                    "PeakSNR_annuli" : self.peaksnr_ann, 
                    "PeakSNR_movement": self.peaksnr_move, 
                    "PeakSNR_kl" : self.peaksnr_kl,
                    "AvgSNR_annuli" : self.avgsnr_ann,
                    "AvgSNR_movement" : self.avgsnr_move, 
                    "AvgSNR_kl" : self.avgsnr_kl,
                    "Contrast_annuli" : self.contrast_ann, 
                    "Contrast_movement" : self.contrast_move, 
                    "Contrast_kl" : self.contrast_kl,
                    "Avg_annuli": self.all_ann,
                    "Avg_move": self.all_move,
                    "Avg_kl": self.all_kl}

        return results

    def collapse_peaksnr(self, data):
        """Collapse the parameter explorer in the peak snr dimension
        Args:
            data(np.array): Array of the parameter explorer file (FITS file data)
        Returns:
            int: annuli value resulting in highest peak snr
            int: movement value resulting in highest peak snr
            int: kl value resulting in highest peak snr
        """
        

        # Find peak SNR best parameters
        collapsed_peaksnr = data[0][0]
        peaksnr_kl, peaksnr_ann, peaksnr_move = np.where(collapsed_peaksnr == np.nanmax(collapsed_peaksnr))
        return peaksnr_ann[0] + 1, peaksnr_move[0], peaksnr_kl[0] # Add one to annuli to make up for python indexing

    def collapse_avgsnr(self, data):
        """Collapse the parameter explorer in the avg snr dimension
        Args:
            data(np.array): Array of the parameter explorer file (FITS file data)
        Returns:
            int: annuli value resulting in highest avg snr
            int: movement value resulting in highest avg snr
            int: kl value resulting in highest avg snr
        """
        
        
        # Find avg SNR best parameters
        collapsed_avgsnr = data[2][0]
        avgsnr_kl, avgsnr_ann, avgsnr_move = np.where(collapsed_avgsnr == np.nanmax(collapsed_avgsnr))
        return avgsnr_ann[0] + 1, avgsnr_move[0], avgsnr_kl[0] # Add one to annuli to make up for python indexing

    def collapse_contrast(self, data):
        """Collapse the parameter explorer in the contrast dimension
        Args:
            data(np.array): Array of the parameter explorer file (FITS file data)
        Returns:
            int: annuli value resulting in highest contrast
            int: movement value resulting in highest contrast
            int: kl value resulting in highest contrast
        """

        # Find contrast best parameters
        collapsed_contrast = -1*data[-1][0]
        log_contrast = np.log10(collapsed_contrast)

        # Filter unphysical contrasts
        log_contrast[log_contrast > 0] = np.nan

        # Take absolute value
        abs_log_contrast = np.abs(log_contrast)

        # Subtract minimum
        abs_log_contrast_sub = abs_log_contrast - np.nanmin(abs_log_contrast)

        # Divide by the max so values go from 0-->1
        contrast = abs_log_contrast_sub / (np.nanmax(abs_log_contrast) - np.nanmin(abs_log_contrast))
        contrast_kl, contrast_ann, contrast_move = np.where(contrast == np.nanmax(contrast))

        return contrast_ann[0] + 1, contrast_move[0], contrast_kl[0]
        
    def collapse_all(self, data):
        """Collapse the parameter explorer across all dimensions
        Args:
            data(np.array): Array of the parameter explorer file (FITS file data)
        Returns:
            int: annuli value resulting in highest average
            int: movement value resulting in highest average
            int: kl value resulting in highest average
        """

        # Collect the pe's for individual collapsed methods
        peaksnr_only = data[0][0] # Peak SNR
        avgsnr_only = data[2][0] # Avg SNR

        collapsed_contrast = -1*data[-1][0] # Contrast
        log_contrast = np.log10(collapsed_contrast)

        # Filter unphysical contrasts
        log_contrast[log_contrast > 0] = np.nan

        # Take absolute value
        abs_log_contrast = np.abs(log_contrast)

        # Subtract minimum
        abs_log_contrast_sub = abs_log_contrast - np.nanmin(abs_log_contrast)

        # Divide by the max so values go from 0-->1
        contrast_only = abs_log_contrast_sub / (np.nanmax(abs_log_contrast) - np.nanmin(abs_log_contrast))

        collapsed_all = np.sum([peaksnr_only, avgsnr_only, contrast_only], axis = 0)

        all_kl, all_ann, all_move = np.where(collapsed_all == np.nanmax(collapsed_all))

        return all_ann[0] + 1, all_move[0], all_kl[0]


    def analyze(self, datadir, annuli, movement):
        """
        Runs KLIP and measures planet SNR given a set of data, parameter combinations, and other KLIP-relevant parameters.
        
        Args:
            datadir (str): Directory containing your data
            ra (list:int): The separation/s (in pixels) of your planet/s from its/their host star
            pa (list:int): The position angle/s (in degrees) of your planet/s from its/their host star
            annuli (int): Annuli value to be tested
            movement (int): Movement value to be tested
            highpass (int): Desired highpass value in pixels (FWHM is recommended)
            iwa (int): Desired inner working angle for KLIP (0 is default)
            outputdir (str): Output directory for KLIP-ed data
            dataname (str): prefix for name of output dataset (ends with '-KLmodes-all.fits')
        """

        mask = (self.ra, self.pa, wid)
        filelist = glob.glob(datadir)

        dataset = MagAO.MagAOData(filelist, highpass=False) 
        dataset.IWA = self.hp * 2
        prefix = f'{self.dataname}_{annuli}_{movement}'
        parallelized.klip_dataset(dataset, outputdir=outputdir, fileprefix=prefix, algo='klip', annuli=annuli, subsections=1, movement=movement,
        numbasis=[1,2,3,4,5,10,20,50,100], calibrate_flux=False, mode="ADI", highpass=highpass, save_aligned=False, time_collapse='median', maxnumbasis=100, verbose = False)

        klipped = np.nanmean(dataset.output[:,:,0,:,:], axis = 1)

        snrmaps, peaksnr, snrsums, snrspurious = snr.create_map(klipped, fwhm = 8, smooth=1, planets=mask, saveOutput=False, sigma = 3, checkmask=False, verbose = True, outputName = 'Test')







def measure_snr(x):
    
    """
    Measures planet SNR given a set of data
    
    Args:
        data (str): Directory containing your data
        ra (list:int): The separation/s (in pixels) of your planet/s from its/their host star
        pa (list:int): The position angle/s (in degrees) of your planet/s from its/their host star
    """
    
    datadir = x[0] 
    ra = x[1]
    pa = x[2] 
    wid = [5,20]
    smooth = 1
    
    mask = (ra, pa, wid)
    snrmaps, peaksnr, snrsums, snrspurious = snr.create_map(datadir, fwhm = 8, smooth=smooth, planets=mask, saveOutput=True, sigma = 3, checkmask=False, verbose = False)

        
    return peaksnr[0]

