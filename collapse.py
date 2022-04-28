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
import panel as pn
from panel.interact import interact
pn.extension()
import hvplot.pandas
import holoviews as hv
from bokeh.models import HoverTool
from holoviews import opts
hv.extension('bokeh')
    

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
        collapsed_peaksnr_norm = collapsed_peaksnr / np.nanmax(collapsed_peaksnr)
        peaksnr_kl, peaksnr_ann, peaksnr_move = np.where(collapsed_peaksnr_norm == np.nanmax(collapsed_peaksnr_norm))
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
        collapsed_avgsnr_norm = collapsed_avgsnr / np.nanmax(collapsed_avgsnr)
        avgsnr_kl, avgsnr_ann, avgsnr_move = np.where(collapsed_avgsnr_norm == np.nanmax(collapsed_avgsnr_norm))
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

        # Normalize
        peaksnr_only_norm = peaksnr_only / np.nanmax(peaksnr_only)
        avgsnr_only_norm = avgsnr_only / np.nanmax(avgsnr_only)

        # Contrast
        collapsed_contrast = -1*data[-1][0] 
        log_contrast = np.log10(collapsed_contrast)

        # Filter unphysical contrasts
        log_contrast[log_contrast > 0] = np.nan

        # Take absolute value
        abs_log_contrast = np.abs(log_contrast)

        # Subtract minimum
        abs_log_contrast_sub = abs_log_contrast - np.nanmin(abs_log_contrast)

        # Divide by the max so values go from 0-->1
        contrast_only_norm = abs_log_contrast_sub / (np.nanmax(abs_log_contrast) - np.nanmin(abs_log_contrast))

        collapsed_all = np.sum([peaksnr_only_norm, avgsnr_only_norm, contrast_only_norm], axis = 0)

        all_kl, all_ann, all_move = np.where(collapsed_all == np.nanmax(collapsed_all))

        return all_ann[0] + 1, all_move[0], all_kl[0]

    def _plot_snrmap(self, snrmap):
        """Make interactive snr map with sliding kl widget"""

        def get_image(frame):
            """Create snrmap image and frame sliding function"""
            snrmap_snr = snrmap[0]
            im = hv.Image(data=snrmap_snr[frame], bounds = (0,0,len(snrmap_snr[1]),len(snrmap_snr[2])))
            tooltips = [("X", "$x"), ("Y", "$y"), ("SNR", "@image")] # Adds hover names and axes
            hover = HoverTool(tooltips=tooltips)
            return im.opts(xlim=(200, 250), 
                ylim=(200, 250), 
                cmap="magma",
                clim=(0, 5),
                xaxis=None,
                yaxis=None,
                tools=[hover],
            )

        # Add frame slider widget
        frame_slider = pn.widgets.DiscreteSlider(
        value=0,
        options={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 10: 5, 25: 6, 50: 7, 100: 8},
        name="KL",
        )

        # Show plot with panel
        @pn.depends(frame=frame_slider)
        def image(frame):
            return get_image(frame)
        img_dmap = hv.DynamicMap(image)

        # Find planet location and draw circle
        dy =  self.ra*np.sin((np.radians(self.pa+90)))
        dx =  self.ra*np.cos((np.radians(self.pa+90)))
        center = int(len(snrmap[0][1])/2)
        x = center + dx
        y = center + dy
        ellipse = hv.Ellipse(x,y,4)

        return pn.Row(img_dmap * ellipse, pn.Column(frame_slider))


    def klip_analyze(self, datadir, annuli, movement, outputdir):
        """
        Runs KLIP and measures planet SNR given a set of data, parameter combinations, and other KLIP-relevant parameters.
        
        Args:
            datadir (str): Directory containing your data
            annuli (int): Annuli value to be tested
            movement (int): Movement value to be tested
            outputdir (str): Output directory for KLIP-ed data
            
        """

        # Make mask of width 'wid' to cover planets when making snr measurements
        wid = [5,10]
        mask = ([self.ra], [self.pa], wid)

        # Get data and set params
        filelist = glob.glob(datadir)
        dataset = MagAO.MagAOData(filelist, highpass=False) 
        dataset.IWA = self.hp * 2
        prefix = f'{self.dataname}_{annuli}_{movement}'

        # Run KLIP
        parallelized.klip_dataset(dataset, outputdir=outputdir, fileprefix=prefix, algo='klip', annuli=annuli, subsections=1, movement=movement,
        numbasis=[1,2,3,4,5,10,20,50,100], calibrate_flux=False, mode="ADI", highpass=self.hp, save_aligned=False, time_collapse='median', maxnumbasis=100, verbose = False)

        # Get klipped data
        klipped = np.nanmean(dataset.output[:,:,0,:,:], axis = 1)

        # Make SNR map
        snrmaps, peaksnr, snrsums, snrspurious = snr.create_map(klipped, fwhm = 8, smooth=1, planets=mask, saveOutput=False, sigma = 3, checkmask=False, verbose = True, outputName=f"{self.dataname}_{annuli}_{movement}_KLModes-all_allsnrmap")
       

        return  self._plot_snrmap(snrmaps)

    def analyze_all(self, datadir, outputdir):
        """Analyze all output best parameters from collapser
        
        Perform a KLIP reduction and create an snrmap using the parameters given as best by the collapser tool
        
        Args:
            datadir (str): Location of data files to be klipped in the form "data/*fits"
            outputdir (str): Location to put output klipped file and snr map e.g. "output/"

        Returns:
            interactive plot of snrmaps
        
        """
        
        # Peak SNR
        map1 = self.klip_analyze(datadir, annuli = self.peaksnr_ann, movement = self.peaksnr_move, outputdir = outputdir)
        
        # Avg SNR
        map2 = self.klip_analyze(datadir, annuli = self.avgsnr_ann, movement = self.avgsnr_move, outputdir = outputdir)
        
        map3 = self.klip_analyze(datadir, annuli = self.contrast_ann, movement = self.contrast_move, outputdir = outputdir)


        

        return map1 + map2 + map3
        
        
        












