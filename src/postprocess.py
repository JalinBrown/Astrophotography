import numpy as np
import sys
from astropy.io import fits
from astropy import units as u
import ccdproc as ccdp
from pathlib import Path
from scipy.ndimage import median_filter, gaussian_filter
from skimage.restoration import denoise_bilateral
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

base_dir = Path("./data")
fits_dir = base_dir / "fits_data/"
stack_output = fits_dir / "stacked" 

input_stack_file = stack_output / "stacked.fits"
output_fits = fits_dir / "processed" / "processed_color.fit"
output_png = fits_dir / "processed" / "processed_color.png"
output_fits.parent.mkdir(parents=True, exist_ok=True)


GRADIENT_MESH_SIZE = 128
STRETCH_MULTIPLIER = 0.070   # Aggressive stretch to lift faint detail.
STAR_REDUCTION_FACTOR = 0.3  # Keeps stars brighter (less reduction).
SATURATION_BOOST = 2.0       # Vibrant color.
BLUE_PUSH = 1.00             
RED_PUSH = 1.00              
BOOST_FACTOR = 0.6           # Local contrast applied to the STRETCHED image.
BLACK_POINT = 1.5            # Crushes the lifted gray background to black.
GAMMA = 0.9                  # Brightens mid-tones (galaxy structure) for final pop.


def remove_gradient(img, mesh=GRADIENT_MESH_SIZE):
    """Dynamically estimate and subtract the background gradient (DBE)."""
    h, w = img.shape
    bg_model = np.zeros_like(img, dtype=np.float32)
    for y in range(0, h, mesh):
        for x in range(0, w, mesh):
            tile = img[y:y+mesh, x:x+mesh]
            bg = np.median(tile)
            bg_model[y:y+mesh, x:x+mesh] = bg
    bg_model = gaussian_filter(bg_model, sigma=mesh/2)
    return img - bg_model


def star_reduction(img, sigma=15, reduction_factor=STAR_REDUCTION_FACTOR):
    """Controls star size. Lower reduction_factor means brighter stars."""
    galaxy_model = gaussian_filter(img, sigma=sigma)
    star_mask = img - galaxy_model
    return img - (star_mask * reduction_factor) 


def run_post_processing():
    print("\n--- Starting Final Color Post-Processing ---")


    print("1. Loading and Demosaicing stacked data.")
    if not input_stack_file.exists():
        print(f"Error: Stacked file not found at {input_stack_file}")
        return

    stacked_ccd = ccdp.CCDData.read(str(input_stack_file), unit=u.adu) # type: ignore
    data = stacked_ccd.data.astype(np.float32)

    # Assumes RGGB pattern for array slicing
    try:
        R = data[::2, ::2]
        G = (data[::2, 1::2] + data[1::2, ::2]) / 2.0
        B = data[1::2, 1::2]
        color_image = np.dstack([R, G, B])
        if color_image.ndim != 3 or color_image.shape[2] != 3:
             raise ValueError("NumPy separation failed.")
             
    except Exception as e:
        print(f"\nCRITICAL ERROR: Demosaicing failed: {e}")
        sys.exit(1)


    print("2. Removing gradients from each color channel.")
    img_detrended_r = remove_gradient(color_image[:, :, 0])
    img_detrended_g = remove_gradient(color_image[:, :, 1])
    img_detrended_b = remove_gradient(color_image[:, :, 2])
    img_detrended_color = np.dstack([img_detrended_r, img_detrended_g, img_detrended_b])



    print("3. Applying noise reduction and star reduction.")
    img_denoised = denoise_bilateral(
        img_detrended_color, sigma_color=0.05, sigma_spatial=2, channel_axis=2 
    )
    img_denoised = median_filter(img_denoised, size=3)

    img_galaxy_dominant = np.zeros_like(img_denoised)
    for i in range(3):
        img_galaxy_dominant[:, :, i] = star_reduction(img_denoised[:, :, i])

    # Clean up negatives before stretching
    img_clean = img_galaxy_dominant - np.min(img_galaxy_dominant)


    print("4. Applying aggressive non-linear stretch (Galaxy POP).")
    stretched = np.arcsinh(img_clean * STRETCH_MULTIPLIER)
    img_contrast_base = stretched
    
    
    print("5. Applying local contrast enhancement.")
    # Now that the image is stretched, we can safely use a higher BOOST_FACTOR
    blurred_low_freq = gaussian_filter(img_contrast_base, sigma=30, axes=2)
    high_freq_detail = img_contrast_base - blurred_low_freq

    img_final_base = img_contrast_base + (high_freq_detail * BOOST_FACTOR)
    img_balanced = img_final_base.copy()
    
    
    print("6. Color balancing.")
    # Color balance applied to the stretched data
    median_g = np.median(img_balanced[:, :, 1])
    r_factor = median_g / np.median(img_balanced[:, :, 0])
    b_factor = median_g / np.median(img_balanced[:, :, 2])

    img_balanced[:, :, 0] *= r_factor * RED_PUSH 
    img_balanced[:, :, 2] *= b_factor * BLUE_PUSH 

    
    print("7. Boosting color saturation.")
    color_boosted = img_balanced.copy() 
    luminance = np.mean(color_boosted, axis=2, keepdims=True)
    color_boosted = luminance + (color_boosted - luminance) * SATURATION_BOOST
    color_boosted = np.clip(color_boosted, 0, None)


    print("8. Adjusting contrast to crush blacks.")
    
    # Clip shadows to absolute zero (DARK BACKGROUND)
    contrast_adjusted = color_boosted - BLACK_POINT
    contrast_adjusted[contrast_adjusted < 0] = 0 
    
    # Rescale to 0-1 and apply Gamma (POP the galaxy)
    max_val = np.max(contrast_adjusted)
    if max_val > 0:
        contrast_adjusted /= max_val
    contrast_adjusted = np.power(contrast_adjusted, GAMMA)
    color_boosted = contrast_adjusted 


    # 9. Save Results
    print("9. Saving results.")
    stretched_norm = rescale_intensity(color_boosted, out_range=(0,1)) # type: ignore
    
    hdu = fits.PrimaryHDU(color_boosted.astype(np.float32))
    hdu.writeto(str(output_fits), overwrite=True)

    plt.imsave(str(output_png), stretched_norm) 

    print("\nColor Post-Processing complete!")

run_post_processing()