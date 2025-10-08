import cv2
import rawpy
import numpy as np
from astropy.io import fits


class ImageIO:
    def load_fits(self, path):
        """_summary_

        Args:
            path (_type_): _description_

        Returns:
            _type_: _description_
        """
        hdul = fits.open(path)
        return hdul


    def load_raw(self, path, bit_16=False):
        """This function loads raw data from a specified path.

        Args:
            path (str): The location from which the raw data should be loaded
            bit_16 (bool, optional): Flag for setting bit type as 16bit. Defaults to False.

        Returns:
            Unknown: data from raw file.
        """

        
        try:
            with rawpy.imread(path) as raw:
                if bit_16:
                    return raw.postprocess(output_bps=16)
                else:
                    return raw.postprocess(output_bps=8)
                    
        except FileNotFoundError as e:
            print(e)
    
    def raw_to_fits(self, raw_data, out_name="image"):
        """This function converts raw data to FITS format

        Args:
            raw_data: Raw data pulled from file.
            out_name (str, optional): Name of output file (no extension). Defaults to "image".
        """
        
        # Flag for setting correct bit type
        bit_16 = True if raw_data.dtype == np.uint16 else False
        
        # Convert to gray scale
        if bit_16:
            gray = np.mean(raw_data, axis=2).astype(np.uint16)
        else:
            gray = np.mean(raw_data, axis=2).astype(np.uint8)
        
        
        fits.writeto(f"{out_name}.fits", gray, overwrite=True)

    def raw_to_tiff(self, raw_data, out_name="image"):
        bgr = cv2.cvtColor(raw_data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_name + ".tiff", bgr)


    # TODO: make fits save to tiff 
    # def fits_to_tiff(self, hdul, out_name="image"):
    #     data = hdul[0].data.astype(np.float32)
        
        
    #     if data.ndim > 2:
    #         data = data[0]
    #         # print(data.ndim)
        
    #     data_min, data_max = np.nanmin(data), np.nanmax(data)
    #     bit_depth = 8 if data_max <= 255 and data_min >= 0 else 16

    #     # === Normalize frames ===
    #     frames = []
    #     for frame in data:
    #         frame = np.nan_to_num(frame)
    #         if bit_depth == 8:
    #             norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    #             frame_out = norm.astype(np.uint8)
    #         else:
    #             norm = cv2.normalize(frame, None, 0, 65535, cv2.NORM_MINMAX)
    #             frame_out = norm.astype(np.uint16)
    #         frames.append(frame_out)