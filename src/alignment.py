from astroalign import register
from astropy.io import fits
from config import *
import numpy as np

aligned_dir = fits_dir / "aligned_lights"
aligned_dir.mkdir(exist_ok=True)

def align_calibrated_frames():
    print("\n--- Aligning Calibrated Frames ---")

    cal_files = sorted(list(calibrated_dir.glob("*.fit")))
    if len(cal_files) < 2:
        print("Need at least two frames to align!")
        return

    # Reference frame
    ref_data = fits.getdata(cal_files[0])
    ref_header = fits.getheader(cal_files[0])

    # Fix endian for reference
    ref_data = np.asarray(ref_data, dtype=ref_data.dtype.newbyteorder('='))

    for f in tqdm(cal_files, desc="Aligning frames"):
        data = fits.getdata(f)

        # Fix endian for each image
        data = np.asarray(data, dtype=data.dtype.newbyteorder('='))

        try:
            aligned, footprint = register(data, ref_data)
        except Exception as e:
            print(f"Alignment failed for {f}: {e}")
            continue

        out_file = aligned_dir / f"aligned_{f.name}"
        fits.writeto(out_file, aligned, ref_header, overwrite=True)

    print("Alignment completed!")
    
align_calibrated_frames()