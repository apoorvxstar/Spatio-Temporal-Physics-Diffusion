import os
import numpy as np
import xarray as xr
import torch  # Required for handling tensors
from PIL import Image

# =========================================================
# 1. EXTRACT IMAGES (Fixed for generic dimensions)
# =========================================================
def xarray_nc_to_images(nc_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"--- Restoring Images from {nc_file} ---")
    
    ds = xr.open_dataset(nc_file)
    
    # Try to find the correct variable name
    if 'image_data' in ds:
        da = ds['image_data']
    else:
        # Fallback: grab the first data variable found
        da = ds[list(ds.data_vars)[0]]

    # --- DYNAMIC DIMENSION DETECTION ---
    # Instead of hardcoding 'instance', we get the first dimension name.
    # da.dims returns a tuple like ('instance', 'height', 'width') or ('time', 'h', 'w')
    batch_dim_name = da.dims[0]
    num_items = da.sizes[batch_dim_name]

    print(f"Detected batch dimension: '{batch_dim_name}'")
    print(f"Extracting {num_items} images...")

    for i in range(num_items):
        # Extract raw numpy array using dynamic indexing
        img_array = da.isel({batch_dim_name: i}).values.astype(np.uint8)
        
        # Determine filename
        # Check if the batch dimension has coordinates (labels) we can use as filenames
        if batch_dim_name in da.coords:
            filename = str(da.coords[batch_dim_name][i].item())
        else:
            filename = f"{i}" # Fallback if no specific labels exist

        # Ensure extension exists
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename += ".png"
        
        save_path = os.path.join(output_folder, filename)
        Image.fromarray(img_array).save(save_path)

    print(f"✅ Images saved to: {output_folder}")


# =========================================================
# 2. EXTRACT LATENTS (Fixed for generic dimensions)
# =========================================================
def xarray_nc_to_latents(nc_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"\n--- Restoring Latents from {nc_file} ---")

    ds = xr.open_dataset(nc_file)

    # Check for likely variable names
    if 'latent_data' in ds:
        da = ds['latent_data']
    elif 'image_data' in ds:
        da = ds['image_data']
    else:
        da = ds[list(ds.data_vars)[0]]

    # --- DYNAMIC DIMENSION DETECTION ---
    batch_dim_name = da.dims[0]
    num_items = da.sizes[batch_dim_name]

    print(f"Detected batch dimension: '{batch_dim_name}'")
    print(f"Extracting {num_items} latents...")

    for i in range(num_items):
        # 1. Get data as numpy array
        latent_np = da.isel({batch_dim_name: i}).values
        
        # 2. Convert back to PyTorch Tensor
        latent_tensor = torch.from_numpy(latent_np)

        # 3. Determine filename
        if batch_dim_name in da.coords:
            filename = str(da.coords[batch_dim_name][i].item())
        else:
            filename = f"{i}"

        # Ensure extension exists
        if not filename.lower().endswith('.pt'):
            filename += ".pt"

        # 4. Save as .pt file
        save_path = os.path.join(output_folder, filename)
        torch.save(latent_tensor, save_path)

    print(f"✅ Latents saved to: {output_folder}")


# =========================================================
# USAGE
# =========================================================
if __name__ == "__main__":
    
    # Path to your NC file inside the docker container
    # nc_file_path = 'Data/5701cc18ee36b30e3a0bf6336ae29038.nc'
    # output_dir = 'ERA5_images'
    #xarray_nc_to_images(nc_file_path, output_dir)

    # for nc file containing latents latents (Uncomment if needed)
    xarray_nc_to_latents('Data/MOSDAC_Latents.nc', 'MOSDAC_images')