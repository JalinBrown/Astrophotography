import cv2
import rawpy
import numpy as np
from astropy.io import fits


class ImageIO:
    def load_fits(self, path):
        hdul = fits.open(path)
        return hdul

    def load_raw(self, path):
        try:
            with rawpy.imread(path) as raw:
                return raw.postprocess()
        except FileNotFoundError as e:
            print(e)

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
    