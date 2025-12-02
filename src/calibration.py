import sys, os
import numpy as np
import rawpy
from astropy.io import fits
from astropy import units as u
from ccdproc import CCDData, ImageFileCollection
import ccdproc as ccdp
from pathlib import Path
from datetime import datetime
# from tqdm import tqdm
from config import *


# ==============================================================================
# PART 1: CR2 TO FITS CONVERSION
# ==============================================================================


def convert_cr2_to_fits(raw_src_dir, fits_dest_dir, image_type):
    """
    Converts all CR2 files in a source directory to FITS files
    in a destination directory using rawpy.
    """
    print(f"\n--- Converting {image_type} CR2s to FITS ---")
    raw_src_dir = Path(raw_src_dir)
    fits_dest_dir = Path(fits_dest_dir)

    # Create the destination directory if it doesn't exist
    fits_dest_dir.mkdir(parents=True, exist_ok=True)

    cr2_files = list(raw_src_dir.glob("*.CR2")) + list(raw_src_dir.glob("*.cr2"))
    if not cr2_files:
        print(f"Warning: No CR2 files found in {raw_src_dir}")
        return

    for cr2_path in tqdm(cr2_files, desc=f"Converting {image_type}"):
        try:
            with rawpy.imread(str(cr2_path)) as raw:
                # Extract the raw, 16-bit sensor data (Bayered)
                # This is what we want for calibration

                raw_image = raw.raw_image_visible.astype(np.uint16)

                # Extract essential metadata
                iso = 200
                exposure = 25
                # timestamp_obj = datetime.fromtimestamp(raw.timestamp)
                # timestamp_iso = timestamp_obj.isoformat()

                # Create a FITS Header
                header = fits.Header()
                header["IMAGETYP"] = (image_type, "Image type")
                header[EXPOSURE_KEY] = (exposure, "Exposure time in seconds")
                header["ISO"] = (iso, "Camera ISO setting")
                # header['DATE-OBS'] = (timestamp_iso, 'Observation start time')

                # Create a FITS HDU (Header Data Unit)
                hdu = fits.PrimaryHDU(data=raw_image, header=header)

                # Write the FITS file
                fits_filename = cr2_path.stem + ".fit"
                hdu.writeto(fits_dest_dir / fits_filename, overwrite=True)

        except Exception as e:
            print(f"\nError converting {cr2_path.name}: {e}")

    print(f"\nCompleted conversion for {image_type}.")


# ==============================================================================
# PART 2: MASTER FRAME CREATION & CALIBRATION
# ==============================================================================


def create_master_bias():
    if MASTER_BIAS_FILE.exists():
        print(f"\nLoading existing Master Bias from: {MASTER_BIAS_FILE.name}")
        return ccdp.CCDData.read(MASTER_BIAS_FILE, unit=u.adu)

    print(
        f"\n--- 1. Creating Master Bias Frame from {len(list(fits_bias_dir.glob('*.fit')))} files ---"
    )
    ifc = ImageFileCollection(fits_bias_dir)

    # Get a list of the FULL PATHS to the bias files
    files_to_combine = ifc.files_filtered(include_path=True)

    if not files_to_combine:
        print(f"Error: No FITS files found in {fits_bias_dir}")
        sys.exit()

    # Combine all 50 bias frames
    master_bias = ccdp.combine(
        files_to_combine,
        method="median",
        unit=u.adu,
        mem_limit=2.5e9,
        sigma_clip=True,
        sigma_clip_low_thresh=5,
        sigma_clip_high_thresh=5,
    )

    master_bias.write(MASTER_BIAS_FILE, overwrite=True)
    print(f"Master Bias saved to: {MASTER_BIAS_FILE.name}")
    return master_bias


def create_master_dark(master_bias):
    if MASTER_DARK_FILE.exists():
        print(f"\nLoading existing Master Dark from: {MASTER_DARK_FILE.name}")
        return ccdp.CCDData.read(MASTER_DARK_FILE, unit=u.adu)

    print(
        f"\n--- 2. Creating Master Dark Frame from {len(list(fits_dark_dir.glob('*.fit')))} files ---"
    )

    if not any(fits_dark_dir.glob("*.fit")):
        print(f"Error: No FITS files found in {fits_dark_dir}.")
        sys.exit()

    ifc = ImageFileCollection(fits_dark_dir, keywords=[EXPOSURE_KEY])

    # First, check the exposure times from the summary
    exp_times = set(ifc.summary[EXPOSURE_KEY])

    if len(exp_times) == 0:
        print(f"\n\nCRITICAL ERROR in 'create_master_dark':")
        print(
            f"Error: No '{EXPOSURE_KEY}' keyword found in any FITS files in {fits_dark_dir}."
        )
        print("This means the 'rawpy' conversion failed to get exposure times.")
        sys.exit()

    if len(exp_times) > 1:
        print(f"\n\nCRITICAL ERROR in 'create_master_dark':")
        print(
            f"Error: Your dark frames have multiple different exposure times: {exp_times}"
        )
        print(
            "A Master Dark can only be created from darks with the *exact same* exposure time."
        )
        sys.exit()

    dark_exposure_time = list(exp_times)[0] * u.s
    print(f"Found dark frames with a single exposure time: {dark_exposure_time}")

    dark_frames_to_combine = []

    # Get full paths and loop explicitly
    files_to_load = ifc.files_filtered(include_path=True)
    for f_path in files_to_load:
        dark_frame = CCDData.read(f_path, unit=u.adu)
        dark_subtracted = ccdp.subtract_bias(dark_frame, master_bias)
        dark_frames_to_combine.append(dark_subtracted)

    master_dark = ccdp.combine(
        dark_frames_to_combine,
        method="median",
        unit=u.adu,
        mem_limit=2.5e9,
        sigma_clip=True,
        sigma_clip_low_thresh=5,
        sigma_clip_high_thresh=5,
    )

    master_dark.header[EXPOSURE_KEY] = (
        dark_exposure_time.value,
        "Exposure time of master dark",
    )
    master_dark.write(MASTER_DARK_FILE, overwrite=True)
    print(f"Master Dark saved to: {MASTER_DARK_FILE.name}")
    return master_dark


def create_master_flat(master_bias, master_dark):
    if MASTER_FLAT_FILE.exists():
        print(f"\nLoading existing Master Dark from: {MASTER_FLAT_FILE.name}")
        return ccdp.CCDData.read(MASTER_FLAT_FILE, unit=u.adu)

    print(
        f"\n--- 3. Creating Master Flat Frame from {len(list(fits_flat_dir.glob('*.fit')))} files ---"
    )
    ifc = ImageFileCollection(fits_flat_dir)

    flat_frames_to_combine = []
    master_dark_exp = master_dark.header[EXPOSURE_KEY] * u.s

    files_to_load = ifc.files_filtered(include_path=True)
    for f_path in files_to_load:
        flat_frame = CCDData.read(f_path, unit=u.adu)
        flat_exposure_time = flat_frame.header[EXPOSURE_KEY] * u.s
        flat_subtracted = ccdp.subtract_bias(flat_frame, master_bias)
        flat_corrected = ccdp.subtract_dark(
            flat_subtracted,
            master_dark,
            dark_exposure=master_dark_exp,
            data_exposure=flat_exposure_time,
            scale=True,
        )
        flat_frames_to_combine.append(flat_corrected)

    master_flat = ccdp.combine(
        flat_frames_to_combine,
        method="median",
        unit=u.adu,
        mem_limit=2.5e9,
        sigma_clip=True,
        sigma_clip_low_thresh=5,
        sigma_clip_high_thresh=5,
    )

    master_flat.data /= np.median(master_flat.data)
    master_flat.write(MASTER_FLAT_FILE, overwrite=True)
    print(f"Master Flat saved to: {MASTER_FLAT_FILE.name}")
    return master_flat


def process_light_frames(master_bias, master_dark, master_flat):
    print(
        f"\n--- 4. Calibrating {len(list(fits_light_dir.glob('*.fit')))} Light Frames ---"
    )
    ifc = ImageFileCollection(fits_light_dir, keywords=[EXPOSURE_KEY])

    calibrated_dir.mkdir(exist_ok=True)

    master_dark_exp = master_dark.header[EXPOSURE_KEY] * u.s

    files_to_load = ifc.files_filtered(include_path=True)
    for f_path in files_to_load:
        raw_light = CCDData.read(f_path, unit=u.adu)
        light_exposure_time = raw_light.header[EXPOSURE_KEY] * u.s

        # 1. Subtract Bias
        calibrated_light = ccdp.subtract_bias(raw_light, master_bias)

        # 2. Subtract Scaled Dark
        calibrated_light = ccdp.subtract_dark(
            calibrated_light,
            master_dark,
            dark_exposure=master_dark_exp,
            data_exposure=light_exposure_time,
            scale=True,
        )

        # 3. Apply Flat-Field Correction
        calibrated_light = ccdp.flat_correct(calibrated_light, master_flat)

        output_filename = f"calibrated_{os.path.basename(f_path)}"
        calibrated_light.write(calibrated_dir / output_filename, overwrite=True)

    print(f"\nCompleted calibration of all light frames.")


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    # Check if the raw directories exist
    if not raw_dir.exists():
        print(f"Error: Raw data directory not found at {raw_dir}")
        print("Please create the directory structure as described.")
        sys.exit()

    # --- RUN PART 1: CONVERSION ---
    # Create all FITS directories first
    for d in [
        fits_bias_dir,
        fits_dark_dir,
        fits_flat_dir,
        fits_light_dir,
        master_dir,
        calibrated_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # # Convert all CR2 files to FITS (Comment out lines below when you add files to your dir)
    convert_cr2_to_fits(raw_dir / 'bias/', fits_bias_dir, 'BIAS')
    convert_cr2_to_fits(raw_dir / 'dark/', fits_dark_dir, 'DARK')
    convert_cr2_to_fits(raw_dir / 'flat/', fits_flat_dir, 'FLAT')
    convert_cr2_to_fits(raw_dir / 'light/', fits_light_dir, 'LIGHT')

    print("\n*** CR2 to FITS conversion complete! ***")

    # --- RUN PART 2: CALIBRATION ---
    print("\n*** Starting calibration process... ***")

    # try:
    # 1. Create Master Calibration Frames
    master_bias = create_master_bias()
    master_dark = create_master_dark(master_bias)
    master_flat = create_master_flat(master_bias, master_dark)

    # 2. Calibrate the Science Frames
    process_light_frames(master_bias, master_dark, master_flat)

    print("\n*** Full data reduction complete! ***")
    print(f"Calibrated light frames are saved in: {calibrated_dir}")

    # except Exception as e:
    #     print(f"\nAn error occurred during calibration: {e}")
