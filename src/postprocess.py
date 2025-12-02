import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter, gaussian_filter
from skimage.restoration import denoise_bilateral
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt


# -----------------------------------------
# 1. Load stacked FITS
# -----------------------------------------
input_file = "data/fits_data/stacked/stacked.fits"
output_fits = "data/fits_data/processed/processed.fit"
output_png = "data/fits_data/processed/processed.png"

image = fits.getdata(input_file).astype(np.float32)


# -----------------------------------------
# 2. Remove Gradients  (Dynamic Background Extraction)
# -----------------------------------------
def remove_gradient(img, mesh=64):
    """Breaks image into tiles, estimates local background, and subtracts it."""
    h, w = img.shape
    bg_model = np.zeros_like(img)

    for y in range(0, h, mesh):
        for x in range(0, w, mesh):
            tile = img[y:y+mesh, x:x+mesh]

            # Background = median of local tile
            bg = np.median(tile)

            bg_model[y:y+mesh, x:x+mesh] = bg

    # Smooth the background model to make it gradual
    bg_model = gaussian_filter(bg_model, sigma=mesh/2)

    # Subtract gradient
    return img - bg_model


img_detrended = remove_gradient(image)


# -----------------------------------------
# 3. Noise Reduction
# -----------------------------------------

# Smooth background noise without killing stars
img_bilateral = denoise_bilateral(
    img_detrended,
    sigma_color=0.05,   # how aggressive the noise smoothing is
    sigma_spatial=2,    # how big the neighborhood is
    channel_axis=None
)

# Light median filter to reduce leftover speckle
img_denoised = median_filter(img_bilateral, size=3)
# -----------------------------------------
# 4. Stretch the image for viewing
# -----------------------------------------

# Remove negative values from gradient subtraction
img_clean = img_denoised - np.min(img_denoised)

# Apply non-linear stretch (arcsinh)
stretched = np.arcsinh(img_clean * 0.005)

# Normalize to 0–1 for PNG
stretched_norm = rescale_intensity(stretched, out_range=(0,1))


# -----------------------------------------
# 5. Save results
# -----------------------------------------
fits.writeto(output_fits, img_clean.astype(np.float32), overwrite=True)
plt.imsave(output_png, stretched_norm, cmap="gray")

print("Processing complete!")
print("Saved:")
print("•", output_fits)
print("•", output_png)