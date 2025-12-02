from pathlib import Path
from tqdm import tqdm
# 1. Define Base Paths
# The script assumes your CR2s are in 'astro_data/cr2_raw'
base_dir = Path("./data")
raw_dir = (
    base_dir / "raw_data"
)  # Create dir for raw photos with folder of the frame types
fits_dir = base_dir / "fits_data/"  # Create dir for fits data to be saved

# 2. Define FITS Directories (Script will create these)
fits_bias_dir = fits_dir / "bias/"
fits_dark_dir = fits_dir / "dark/"
fits_flat_dir = fits_dir / "flat/"
fits_light_dir = fits_dir / "light/"
master_dir = fits_dir / "master_frames"
calibrated_dir = fits_dir / "calibrated_lights"

# 3. Define FITS Header Keyword for Exposure Time
# We set this to 'EXPTIME'. The conversion step (Part 1) will
# create this keyword, and the calibration step (Part 2) will read it.
EXPOSURE_KEY = "EXPTIME"

# Naming convention for the final master frames
MASTER_BIAS_FILE = master_dir / "MasterBias.fit"
MASTER_DARK_FILE = master_dir / "MasterDark.fit"
MASTER_FLAT_FILE = master_dir / "MasterFlat.fit"


aligned_dir = fits_dir / "aligned_lights"
aligned_dir.mkdir(exist_ok=True)

stack_output = fits_dir / "stacked"
stack_output.mkdir(exist_ok=True)