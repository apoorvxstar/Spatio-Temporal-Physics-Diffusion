================================================================================
             SPATIO-TEMPORAL PHYSICS-INFORMED DIFFUSION MODEL (STPD)
                    Training, Deployment & Execution Guide
================================================================================
PROJECT:   Cloud Pattern Prediction (PiSTLDM)
AUTHOR:    Manav Vashi, Apoorv Patel (Intern, CAIR - DRDO)
PLATFORM:  Docker / Python 3.11 / NVIDIA GPU
================================================================================

*** IMPORTANT NAMING CONVENTION ***
To distinguish between the two core modules, please observe the following:
  1. CONVOLUTION AUTOENCODER (CAE): Files use all lowercase letters.
     (e.g., train.py, model.py)
  2. DIFFUSION MODEL: Files use Capitalized letters.
     (e.g., Train.py, Model.py)

*** DIRECTORY STRUCTURE ***
  - /Python_files/Training/CAE_training       : Scripts for CAE
  - /Python_files/Training/Diffusion_training : Scripts for Diffusion
  - /Python_files/ncFile_Handling             : Data conversion (NetCDF)
  - /Python_files/Latent_Generation           : Latent space encoding/decoding

================================================================================
                            WORKFLOW & COMMANDS
================================================================================
NOTE: Ensure you are in the root directory (Cloud_Prediction_STPD) before 
running these commands. All scripts run inside the Docker container.

--------------------------------------------------------------------------------
1. LOAD DOCKER IMAGE
--------------------------------------------------------------------------------
Load the pre-built Docker image from the diffusion_py311.tar file into your local Docker 
environment.

COMMAND:
sudo docker load -i diffusion_py311.tar

--------------------------------------------------------------------------------
2. CHECK AVAILABLE IMAGES
--------------------------------------------------------------------------------
Confirm that 'diffusion:py311' is listed in your local Docker repository.

COMMAND:
sudo docker images

--------------------------------------------------------------------------------
3. CHECK DEPENDENCIES INSIDE DOCKER IMAGE
--------------------------------------------------------------------------------
Verify that the correct Python libraries are installed within the container.

COMMAND:
sudo docker run --rm diffusion:py311 pip list

--------------------------------------------------------------------------------
4. REMOVE DOCKER IMAGE (Maintenance)
--------------------------------------------------------------------------------
Use this only if you need to clean up space or reload a fresh version of the 
image.

COMMAND:
sudo docker rmi diffusion:py311

--------------------------------------------------------------------------------
5. NC FILE CONVERSION (Preprocessing)
--------------------------------------------------------------------------------
Convert raw data (images/latents) into .nc (NetCDF) format for processing.

COMMAND:
sudo docker run --rm --gpus all -v "$(pwd):/app" -w /app diffusion:py311 python Python_files/ncFile_Handling/nc_generator.py

--------------------------------------------------------------------------------
6. NC REVERSE (Verification)
--------------------------------------------------------------------------------
Convert .nc data back into images or latents to verify data integrity.

COMMAND:
sudo docker run --rm --gpus all -v "$(pwd):/app" -w /app diffusion:py311 python Python_files/ncFile_Handling/nc_reverse.py

--------------------------------------------------------------------------------
7. CAE TRAINING (Convolution Autoencoder)
--------------------------------------------------------------------------------
Train the Autoencoder to compress cloud frames into latent representations.
(Files located in: Python_files/Training/CAE_training)

COMMAND:
sudo docker run --rm --gpus all -v "$(pwd):/app" -w /app diffusion:py311 python Python_files/Training/CAE_training/train.py

--------------------------------------------------------------------------------
8. IMAGES TO LATENT CREATION
--------------------------------------------------------------------------------
Use the trained CAE to compress the entire dataset of images into latent vectors.

COMMAND:
sudo docker run --rm --gpus all -v "$(pwd):/app" -w /app diffusion:py311 python Python_files/Latent_Generation/image_to_latent.py

--------------------------------------------------------------------------------
9. LATENT TO IMAGE CREATION (Reconstruction Check)
--------------------------------------------------------------------------------
Decode latent vectors back into images to check the quality of the CAE reconstruction.

COMMAND:
sudo docker run --rm --gpus all -v "$(pwd):/app" -w /app diffusion:py311 python Python_files/Latent_Generation/latent_to_image.py

--------------------------------------------------------------------------------
10. DIFFUSION TRAINING (PiSTLDM)
--------------------------------------------------------------------------------
Train the core Diffusion model on the latent vectors created in Step 8.
(Files located in: Python_files/Training/Diffusion_training)

COMMAND:
sudo docker run --rm --gpus all -v "$(pwd):/app" -w /app diffusion:py311 python Python_files/Training/Diffusion_training/Train.py

--------------------------------------------------------------------------------
11. RUN STREAMLIT UI (Inference/Demo)
--------------------------------------------------------------------------------
Launch the interactive web interface to generate cloud predictions.
Access at: http://localhost:8501

COMMAND:
sudo docker run -it --rm \
   --gpus all \
   -p 8501:8501 \
   -v "$(pwd)":/workspace \
   -w /workspace \
   diffusion:py311 \
   streamlit run app.py --server.address=0.0.0.0 --server.port=8501

================================================================================
                           COMMAND GLOSSARY
================================================================================
--rm                    : Automatically remove container when it exits.
--gpus all              : Grant container access to NVIDIA GPUs.
-v "$(pwd):/app"        : Mount current directory to /app inside container.
-w /app                 : Set working directory inside container to /app.
-p 8501:8501            : Map port 8501 (Host) to 8501 (Container).
-it                     : Interactive mode (view logs in real-time).
