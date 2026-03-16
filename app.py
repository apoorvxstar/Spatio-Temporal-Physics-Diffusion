# app.py
import streamlit as st
import os
import torch
import tempfile
import io
import zipfile
from datetime import datetime  # <--- Added for timestamping
from inference import InferenceConfig, run_inference_from_images, ProgressCallback
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import numpy as np

import DConfig as org

os.environ["TORCH_HOME"] = os.path.abspath("weights/torch_hub")
os.environ["HF_HUB_OFFLINE"] = "1"

st.set_page_config(layout="wide")

# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
<style>
/* Sidebar Adjustments */
[data-testid="stSidebarHeader"] {height: 0.5rem;}
.stSidebar > div:first-child {padding-top: 1rem;}
.block-container {padding-top: 1.5rem; padding-bottom: 1rem;}

/* File Uploader Cleanups */
[data-testid="stFileUploaderDropzone"] svg {display: none;}
[data-testid="stFileUploaderDropzone"] {min-height: auto; height: auto; padding: 0.5rem !important;}
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {display: none;}
[data-testid="stFileUploaderDropzoneInstructions"] {display: block !important; text-align: center !important;}
[data-testid="stFileUploaderDropzoneInstructions"]::before {content: "Upload 4 past frames + 1 true frame";}
[data-testid="stFileUploaderDropzoneInstructions"] > * {display: none;}
.stFileUploaderFile {display: none;}
[data-testid="stFileUploaderPagination"] {display: none;}

/* SQUARE CORNERS ENFORCEMENT */
img { border-radius: 0px !important; }
[data-testid="stImage"] > img { border-radius: 0px !important; }

/* RESET BUTTON STYLE */
div.stButton > button:first-child {
    width: 100%;
}           
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER & RESET BUTTON
# ==========================
col_header, col_reset = st.columns([8, 1], vertical_alignment="bottom")

with col_header:
    st.markdown("<h2 style='text-align: center; margin-bottom: 10px;'>Spatio-Temporal Physics Informed Diffusion</h2>", unsafe_allow_html=True)

with col_reset:
    if st.button("Reset", type="primary", help="Reset the application"):
        st.session_state.clear()
        st.rerun()

DIFF_T = int(org.diff_timesteps)

# ==========================
# STREAMLIT PROGRESS
# ==========================
class StreamlitProgress(ProgressCallback):
    def __init__(self):
        self.bar = st.progress(0)
        self.text = st.empty()

    def update(self, percent, message):
        self.bar.progress(min(int(percent), 100))
        self.text.markdown(f"**{message}**")

# ==========================
# SIDEBAR CONTROLS
# ==========================
st.sidebar.header("Inference Settings")

if "physics_guidance" not in st.session_state:
    st.session_state["physics_guidance"] = True

apply_pics = st.sidebar.toggle("Physics Guidance", key="physics_guidance")

if apply_pics:
    guidance_peak_scale = st.sidebar.number_input(
        "Guidance Peak Scale",
        min_value=0.0, max_value=1.0, value=0.005, step=0.0005, format="%.4f",
        help="Maximum guidance weight for physics consistency"
    )

    guidance_min_scale = st.sidebar.number_input(
        "Guidance Min Scale",
        min_value=0.0, max_value=1.0, value=0.0005, step=0.0005, format="%.4f",
        help="Minimum guidance weight applied during diffusion"
    )

    guidance_peak_step = st.sidebar.number_input(
        "Guidance Peak Step",
        min_value=1, max_value=DIFF_T, value=500, step=1,
        help="Diffusion step where guidance reaches its peak"
    )

    guidance_width = st.sidebar.number_input(
        "Guidance Width",
        min_value=100, max_value=DIFF_T, value=200, step=100,
        help="Width of the guidance Gaussian curve"
    )

    guidance_interval = st.sidebar.number_input(
        "Guidance Interval",
        min_value=1, max_value=1000, value=10, step=1,
        help="Apply guidance every N steps"
    )
else:
    guidance_peak_scale = 0.0
    guidance_min_scale = 0.0
    guidance_peak_step = 0 
    guidance_width = 0
    guidance_interval = 0
    st.sidebar.info("Physics guidance is disabled.")

vis_freq = 25 

cfg = InferenceConfig(
    guidance_peak_scale=guidance_peak_scale,
    guidance_min_scale=guidance_min_scale,
    guidance_peak_step=guidance_peak_step,
    guidance_width=guidance_width,
    guidance_interval=guidance_interval,
    vis_freq=vis_freq 
)

# ==========================
# SESSION STATE INITIALIZATION
# ==========================
if "inference_results" not in st.session_state:
    st.session_state.inference_results = None

if "uploaded_image_data" not in st.session_state:
    st.session_state.uploaded_image_data = None

if "processing" not in st.session_state:
    st.session_state.processing = False

# ==========================
# IMAGE UPLOAD LOGIC
# ==========================
if st.session_state.uploaded_image_data is None:
    
    uploaded_files = st.file_uploader(
        label="Upload images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.info("**Note:** Files are processed in alphabetical order (e.g., `01.png`, `02.png`...)")

    if uploaded_files:
        sorted_files = sorted(uploaded_files, key=lambda x: x.name)
        num_files = len(sorted_files)
        
        if num_files == 5:
            try:
                processed_data = []
                first_size = None
                
                for f in sorted_files:
                    bytes_data = f.getvalue()
                    img = Image.open(io.BytesIO(bytes_data)).convert("L")
                    
                    if first_size is None:
                        first_size = img.size
                    elif img.size != first_size:
                        st.error(f"Dimension Mismatch: Image '{f.name}' ({img.size}) does not match previous images.")
                        st.stop()
                    
                    processed_data.append({"name": f.name, "bytes": bytes_data, "size": img.size})
                
                st.session_state.uploaded_image_data = processed_data
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing images: {e}")
                
        elif num_files > 0:
            st.warning(f"You have uploaded {num_files} images. Please upload exactly 5 to proceed.")
else:
    # ==========================
    # PREVIEW
    # ==========================
    image_data_list = st.session_state.uploaded_image_data
    
    st.markdown("### Input Frames Preview")
    cols = st.columns(4)
    for i in range(4):
        with cols[i]:
            img = Image.open(io.BytesIO(image_data_list[i]["bytes"]))
            img = img.resize((512, 512), Image.NEAREST)
            st.image(img, caption=f"Input Frame {i+1}", use_container_width=True)

    # ==========================
    # INFERENCE LOGIC
    # ==========================
    if st.session_state.inference_results is not None:
        pass
    elif st.session_state.processing:
        progress = StreamlitProgress()
        anim_container = st.columns([1, 2, 1])[1]  
        anim_placeholder = anim_container.empty()

        def update_anim(img_tensor):
            img_np = (img_tensor[0, 0].detach().cpu().numpy() + 1) / 2
            img_np = np.clip(img_np, 0, 1)
            pil_img = Image.fromarray((img_np * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)
            
            anim_placeholder.image(
                pil_img,
                caption="Denoising in progress...",
                use_container_width=True
            )

        temp_paths = []
        try:
            for item in image_data_list:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp.write(item["bytes"])
                tmp.close()
                temp_paths.append(tmp.name)

            with st.spinner("Running physics-guided diffusion inference..."):
                past_imgs, pred, gt, metrics, diff_map = run_inference_from_images(
                    cfg, 
                    temp_paths, 
                    progress, 
                    image_callback=update_anim
                )
                
                past_imgs_cpu = [p.cpu() for p in past_imgs]
                pred_cpu = pred.cpu()
                gt_cpu = gt.cpu()
                diff_map_cpu = diff_map.cpu()
                
                st.session_state.inference_results = {
                    "past_imgs": past_imgs_cpu,
                    "pred": pred_cpu,
                    "gt": gt_cpu,
                    "metrics": metrics,
                    "diff_map": diff_map_cpu
                }

            st.session_state.processing = False
            progress.update(100, "Inference completed.")
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.processing = False 
        
        finally:
            for path in temp_paths:
                if os.path.exists(path):
                    try: os.remove(path)
                    except: pass

    else:
        if st.button("Run Inference", use_container_width=True):
            st.session_state.processing = True
            st.rerun()

# ==========================
# DISPLAY RESULTS
# ==========================
if st.session_state.inference_results is not None:
    
    # --- DATE & TIME LOGIC ---
    now = datetime.now()
    # Using hyphens instead of | or : as they are generally safer for filenames across OS
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H-%M-%S")
    
    # Construct suffix: Date-DD-MM-YYYY_Time-HH-MM-SS
    file_suffix = f"Date-{date_str}_Time-{time_str}"
    
    # --- DYNAMIC HEADER TITLE LOGIC ---
    # We check if guidance_peak_scale > 0 to determine if physics guidance was used
    is_guided = guidance_peak_scale > 0
    
    if is_guided:
        header_title = "Physics-Informed Diffusion Inference"
        file_tag = "Physics_Guided"
    else:
        header_title = "Standard Diffusion Inference"
        file_tag = "Standard"

    res = st.session_state.inference_results
    past_imgs = res["past_imgs"]
    pred = res["pred"]
    gt = res["gt"]
    metrics = res["metrics"]
    diff_map = res["diff_map"]
    
    mse, mae, ssim, ms_ssim, lpips = metrics

    # ==========================
    # TABS
    # ==========================
    tab_metrics, tab_frames, tab_diff = st.tabs([
        "Metrics & Report",
        "Input & Predicted Frames",
        "Ground Truth vs Prediction vs Diff"
    ])

    # --------------------------
    # 1. Metrics Tab
    # --------------------------
    with tab_metrics:
        metrics_dict = {
            "MSE": f"{mse:.4f}", "MAE": f"{mae:.4f}", 
            "SSIM": f"{ssim:.4f}", "MS-SSIM": f"{ms_ssim:.4f}", "LPIPS": f"{lpips:.4f}"
        }

        settings_dict = {
            "Guidance Peak Scale": f"{guidance_peak_scale:.4f}",
            "Guidance Min Scale": f"{guidance_min_scale:.4f}",
            "Guidance Peak Step": f"{guidance_peak_step}",
            "Guidance Width": f"{guidance_width}",
            "Guidance Interval": f"{guidance_interval}",
            "Vis Frequency": f"{vis_freq}"
        }

        # UPDATED: Accepts 'title' argument for the image header
        def create_full_report(gt_arr, pred_arr, mets, sets, title):
            fig = plt.figure(figsize=(10, 12))
            
            # Dynamic Header in the Image
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.96)

            ax_gt = plt.subplot2grid((3, 2), (0, 0))
            ax_pred = plt.subplot2grid((3, 2), (0, 1))

            ax_gt.imshow(gt_arr, cmap="gray", aspect='equal')
            ax_gt.set_title("Ground Truth", fontsize=12)
            ax_gt.axis("off")

            ax_pred.imshow(pred_arr, cmap="gray", aspect='equal')
            ax_pred.set_title("Predicted Frame", fontsize=12)
            ax_pred.axis("off")

            ax_text = plt.subplot2grid((3, 2), (1, 0), colspan=2, rowspan=2)
            ax_text.axis("off")

            text_str = (
                "INFERENCE REPORT\n"
                + "=" * 55 + "\n\n"
                "EVALUATION METRICS\n"
                + "-" * 30 + "\n"
            )
            for k, v in mets.items():
                text_str += f"{k:<15}: {v}\n"

            text_str += "\nINFERENCE SETTINGS\n" + "-" * 30 + "\n"
            for k, v in sets.items():
                text_str += f"{k:<25}: {v}\n"

            ax_text.text(
                0.05, 0.95, text_str,
                fontsize=11,
                family="monospace",
                va="top"
            )

            plt.subplots_adjust(top=0.92)
            
            buf_rep = io.BytesIO()
            plt.savefig(buf_rep, format="png", dpi=200)
            buf_rep.seek(0)
            plt.close(fig)
            return buf_rep

        pred_numpy = (pred[0, 0].detach().cpu().numpy() + 1) / 2
        gt_numpy = (gt[0, 0].detach().cpu().numpy() + 1) / 2
        
        # Create report with dynamic title
        report_buffer = create_full_report(gt_numpy, pred_numpy, metrics_dict, settings_dict, header_title)

        col_header_rep, col_button_rep = st.columns([3, 1])
        with col_header_rep: st.subheader("Inference Overview")
        with col_button_rep:
            # UPDATED: File name format
            st.download_button(
                "Download Report", 
                data=report_buffer, 
                file_name=f"Report_{file_tag}_{file_suffix}.png", 
                mime="image/png", 
                use_container_width=True
            )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("MSE", f"{mse:.4f}"); m2.metric("MAE", f"{mae:.4f}"); m3.metric("SSIM", f"{ssim:.4f}")
        m4.metric("MS-SSIM", f"{ms_ssim:.4f}"); m5.metric("LPIPS", f"{lpips:.4f}")

        col_spacer1, col_gt, col_pred, col_spacer2 = st.columns([1, 2, 2, 1])

        gt_pil = Image.fromarray((gt_numpy * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)
        pred_pil = Image.fromarray((pred_numpy * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)

        with col_gt:
            st.image(gt_pil, caption="Ground Truth", clamp=True, use_container_width=True)

        with col_pred:
            st.image(pred_pil, caption="Predicted Frame", clamp=True, use_container_width=True)


    # --------------------------
    # 2. Frames Tab
    # --------------------------
    with tab_frames:
        all_frames_tensors = past_imgs + [pred] 
        frames_pil = [
            Image.fromarray(((img[0, 0].detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8))
            .resize((512, 512), Image.NEAREST) 
            for img in all_frames_tensors
        ]

        total_width = sum(i.width for i in frames_pil)
        max_height = max(i.height for i in frames_pil)
        stitched_img = Image.new('L', (total_width, max_height))
        x_offset = 0
        for im in frames_pil:
            stitched_img.paste(im, (x_offset, 0))
            x_offset += im.width
        
        buf_stitched = io.BytesIO()
        stitched_img.save(buf_stitched, format='PNG')
        buf_stitched.seek(0)

        buf_gif = io.BytesIO()
        frames_pil[0].save(buf_gif, format='GIF', save_all=True, append_images=frames_pil[1:], duration=500, loop=0)
        buf_gif.seek(0)

        c_h1, c_b1 = st.columns([2, 1])
        with c_h1: st.subheader("Input Frames + Predicted Frame")
        with c_b1:
            # UPDATED: File name format
            st.download_button(
                "Download Image", 
                data=buf_stitched, 
                file_name=f"Sequence_{file_tag}_{file_suffix}.png", 
                mime="image/png", 
                use_container_width=True
            )
        
        cols_frames = st.columns(5)
        for i, img_pil in enumerate(frames_pil):
            with cols_frames[i]:
                caption_text = f"Frame {i+1}" if i < 4 else "Predicted"
                st.image(img_pil, caption=caption_text, use_container_width=True, clamp=True)

        c_h2, c_b2 = st.columns([2, 1])
        with c_h2: st.subheader("Denosing Process Animation")
        with c_b2:
            # UPDATED: File name format
            st.download_button(
                "Download Animation", 
                data=buf_gif, 
                file_name=f"Animation_{file_tag}_{file_suffix}.gif", 
                mime="image/gif", 
                use_container_width=True
            )
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2: st.image(buf_gif, caption="Animation Preview", use_container_width=True)

    # --------------------------
    # 3. Difference Map Tab
    # --------------------------
    with tab_diff:
        st.subheader("Comparison View")

        c_control, c_down = st.columns([3, 1], vertical_alignment="bottom")
        
        with c_control:
            cmap_option = st.selectbox(
                "Select Color Scheme", 
                ["binary", "gray", "twilight", "inferno", "seismic"], 
                index=0
            )

        diff_scale = 1.0
        gt_arr = (gt[0, 0].detach().cpu().numpy() + 1) / 2
        pred_arr = (pred[0, 0].detach().cpu().numpy() + 1) / 2
        diff_arr = diff_map[0, 0].detach().cpu().numpy()

        # UPDATED: Accepts 'title' argument
        def generate_aligned_figure(gt_img, pred_img, diff_img, cmap, d_scale, title):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Dynamic Header in the Image
            fig.suptitle(title, fontsize=18, fontweight='bold', y=0.96)

            plots = [
                {"data": gt_img,   "title": "Ground Truth",     "cmap": "gray",  "cbar": False},
                {"data": pred_img, "title": "Predicted",        "cmap": "gray",  "cbar": False},
                {"data": diff_img, "title": f"Difference Map",  "cmap": cmap,    "cbar": True, "vmax": d_scale}
            ]

            for ax, p in zip(axes, plots):
                im = ax.imshow(p["data"], cmap=p["cmap"], vmin=0, vmax=p.get("vmax", 1.0), aspect='equal')
                ax.set_title(p["title"], fontsize=12, pad=10)
                ax.axis("off")

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)

                if p["cbar"]:
                    cbar = fig.colorbar(im, cax=cax)
                    cbar.set_label('Difference Magnitude', rotation=270, labelpad=15)
                else:
                    cax.axis("off")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            return fig

        # Generate with dynamic title
        fig_comparison = generate_aligned_figure(gt_arr, pred_arr, diff_arr, cmap_option, diff_scale, header_title)

        buf_comp = io.BytesIO()
        fig_comparison.savefig(buf_comp, format='png', dpi=300, bbox_inches='tight')
        buf_comp.seek(0)

        st.pyplot(fig_comparison, use_container_width=True)

        with c_down:
            # UPDATED: File name format
            st.download_button(
                label="Download Comparison",
                data=buf_comp,
                file_name=f"Comparison_{file_tag}_{file_suffix}.png",
                mime="image/png",
                use_container_width=True
            )