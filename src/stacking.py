from astropy.stats import sigma_clip
from config import *
import numpy as np
from astropy.io import fits


def stack_aligned_frames():
    print("\n--- Stacking Aligned Frames ---")
    
    aligned_files = sorted(list(aligned_dir.glob("*.fit")))
    stack_list = []

    for f in aligned_files:
        img = fits.getdata(f).astype(float)
        stack_list.append(img)

    stack_array = np.array(stack_list)

    # Sigma-clipped mean
    clipped = sigma_clip(stack_array, sigma=3, axis=0)
    stacked_image = np.mean(clipped.data, axis=0)

    output_file = stack_output / "stacked.fits"
    fits.writeto(output_file, stacked_image, overwrite=True)

    print(f"Stack saved to: {output_file}")

stack_aligned_frames()