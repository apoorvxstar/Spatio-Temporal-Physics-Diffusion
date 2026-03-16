import os
import numpy as np
import xarray as xr
from PIL import Image
import torch  # Required for loading .pt files
from multiprocessing import Pool, cpu_count

# =========================================================
# CONFIG
# =========================================================
NUM_WORKERS = 8
VALID_IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tiff')
VALID_LATENT_EXTS = ('.pt',)

# =========================================================
# WORKER FUNCTIONS (MUST BE TOP-LEVEL FOR WINDOWS)
# =========================================================

def load_single_image(args):
    """Worker to load and process a single image file."""
    image_folder, fname = args
    path = os.path.join(image_folder, fname)

    try:
        with Image.open(path) as img:
            # Convert to true grayscale
            img = img.convert("L")
            arr = np.array(img, dtype=np.uint8)
        return fname, arr

    except Exception as e:
        print(f"Skipping image {fname}: {e}")
        return None

def load_single_latent(args):
    """Worker to load and process a single .pt latent file."""
    latent_folder, fname = args
    path = os.path.join(latent_folder, fname)

    try:
        # Load tensor safely on CPU
        tensor = torch.load(path, map_location='cpu')
        
        # Ensure it's a numpy array
        # If tensor has gradients, detach first
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.array(tensor)

        # Optional: consistency check
        # If your latents were saved as [1, 4, 64, 64], you might want to squeeze
        # if arr.ndim == 4 and arr.shape[0] == 1:
        #     arr = arr.squeeze(0)
            
        return fname, arr

    except Exception as e:
        print(f"Skipping latent {fname}: {e}")
        return None


# =========================================================
# CONVERSION FUNCTIONS
# =========================================================

def images_to_xarray_nc(image_folder, output_nc_file):
    print(f"\n--- Processing Images from: {image_folder} ---")
    
    # 1. Collect files
    file_list = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(VALID_IMG_EXTS)
    ])

    if not file_list:
        print("No images found.")
        return

    print(f"Found {len(file_list)} images")
    
    # 2. Parallel loading
    with Pool(processes=min(NUM_WORKERS, cpu_count())) as pool:
        results = pool.map(
            load_single_image,
            [(image_folder, fname) for fname in file_list]
        )

    # 3. Filter valid results
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("No valid images loaded.")
        return

    valid_filenames, img_data_list = zip(*valid_results)

    # 4. Stack images -> Shape: (instance, height, width)
    stacked_data = np.stack(img_data_list, axis=0)
    print("Stacked image shape:", stacked_data.shape)

    # 5. Create Xarray DataArray
    da = xr.DataArray(
        data=stacked_data,
        dims=["instance", "height", "width"],
        coords={"instance": list(valid_filenames)},
        name="image_data",
        attrs={
            "description": "Grayscale image stack",
            "channels": 1,
            "dtype": str(stacked_data.dtype)
        }
    )

    # 6. Save NetCDF
    encoding = {"image_data": {"zlib": True, "complevel": 5}}
    da.to_netcdf(output_nc_file, encoding=encoding)
    print(f"Saved Images NetCDF -> {output_nc_file}")


def latents_to_xarray_nc(latent_folder, output_nc_file):
    print(f"\n--- Processing Latents from: {latent_folder} ---")
    
    # 1. Collect files
    file_list = sorted([
        f for f in os.listdir(latent_folder)
        if f.lower().endswith(VALID_LATENT_EXTS)
    ])

    if not file_list:
        print("No .pt files found.")
        return

    print(f"Found {len(file_list)} latent files")

    # 2. Parallel loading
    with Pool(processes=min(NUM_WORKERS, cpu_count())) as pool:
        results = pool.map(
            load_single_latent,
            [(latent_folder, fname) for fname in file_list]
        )

    # 3. Filter valid results
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("No valid latents loaded.")
        return

    valid_filenames, latent_data_list = zip(*valid_results)

    # 4. Stack latents
    # Expected Shape: (instance, channels, height, width) 
    # e.g., (1000, 4, 64, 64)
    stacked_data = np.stack(latent_data_list, axis=0)
    print("Stacked latent shape:", stacked_data.shape)

    # 5. Create Xarray DataArray
    # Note: Adjust dims if your latents are 1D vectors or different shapes
    if stacked_data.ndim == 4:
        dims = ["instance", "channel", "height", "width"]
    elif stacked_data.ndim == 2:
        dims = ["instance", "feature"]
    else:
        # Fallback for generic shapes
        dims = ["instance"] + [f"dim_{i}" for i in range(stacked_data.ndim - 1)]

    da = xr.DataArray(
        data=stacked_data,
        dims=dims,
        coords={"instance": list(valid_filenames)},
        name="latent_data",
        attrs={
            "description": "Latent vector stack",
            "dtype": str(stacked_data.dtype)
        }
    )

    # 6. Save NetCDF
    encoding = {"latent_data": {"zlib": True, "complevel": 5}}
    da.to_netcdf(output_nc_file, encoding=encoding)
    print(f"Saved Latents NetCDF -> {output_nc_file}")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    
    # #1. Convert Images to nc
    # images_to_xarray_nc(
    #     "Data/JUL_TIR1_png/dummy_class",
    #     "ERA5_Images.nc"
    # )

    # 2. Convert Latents to nc
    latents_to_xarray_nc(
        "Data/LATENTS_MOSDAC/mosdac_tir1",    
        "MOSDAC_Latents.nc"      
    )