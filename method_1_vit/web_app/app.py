import streamlit as st
from src.preprocessing import prepare_image
from src.analyzers import run_fft, run_ela, run_resnet, run_siglip, run_sdxl_detector, run_deepfake_detector
from src.ensemble import calculate_final_verdict

st.set_page_config(page_title="AI Image Detector", layout="centered", page_icon="👁️")

# Inject Custom CSS for aesthetics
st.markdown("""
    <style>
    /* Make the title look sleeker */
    .stApp > header {
        background-color: transparent;
    }
    
    /* Round corners on all images and limit max height */
    img {
        border-radius: 12px;
        max-height: 400px;
        object-fit: contain;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Style the metrics nice and centered */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #00ff88;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #f0f2f6; font-weight: 800;'>👁️ AI Image Detector Ensemble</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Methodology")
    st.write('''
    This tool uses a 6-method ensemble approach to detect AI-generated images vs. real images:
    
    1. **FFT (Fast Fourier Transform):** Analyzes the image in the frequency domain to find repeating artifacts common in AI generation (like checkerboard patterns).
    2. **ELA (Error Level Analysis):** Highlights areas of an image that are compressed at different quality levels, which can indicate tampering or non-uniform synthesis.
    3. **ResNet Model:** Uses `umm-maybe/AI-image-detector`.
    4. **SigLIP Model:** Uses `Ateeqq/ai-vs-human-image-detector`.
    5. **SDXL Detector:** Uses `Organika/sdxl-detector` to catch modern diffusion artifacts.
    6. **DeepFake Detector (ViT):** Uses `prithivMLmods/Deep-Fake-Detector-v2-Model` for state-of-the-art vision transformer detection.
    
    **The Final Judge:** The final verdict averages all 4 deep-learning AI model scores.
    ''')
    st.markdown("---")
    st.write("Developed for transparency in AI generation.")

uploaded_file = st.file_uploader("Upload an image to analyze", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    st.markdown("---")
    
    with st.spinner("The Janitor: Validating, stripping metadata, & preparing artifacts..."):
        # Preprocessing
        grayscale_array, ela_jpeg_img, rgb_img = prepare_image(uploaded_file)
        
    # Display the cleaned image
    st.image(rgb_img, caption="Ready for Analysis", use_column_width="always")
    
    st.markdown("---")
    st.header("The Final Judge")
    final_judge_placeholder = st.empty()
    final_judge_placeholder.info("⏳ Waiting for all engines to complete analysis...")
    st.markdown("---")
    
    # Analysis row 1: Visual Techniques (FFT & ELA)
    st.header("Visual and Frequency Artifacts")
    col1, col2 = st.columns(2)
    
    with col1:
        with st.spinner("Engine 1: Running Fast Fourier Transform..."):
            fft_fig = run_fft(grayscale_array)
        st.subheader("FFT Analysis")
        st.pyplot(fft_fig)
        st.info("💡 **How to read FFT:** Real photos usually have a bright center that smoothly fades outwards. AI images often show harsh geometric patterns, grids, or bright stars radiating from the center due to the synthesis process.")
        
    with col2:
        with st.spinner("Engine 2: Running Error Level Analysis..."):
            ela_img = run_ela(rgb_img, ela_jpeg_img)
        st.subheader("Error Level Analysis")
        st.image(ela_img, caption="Enhanced ELA Difference", use_column_width="always")
        st.info("💡 **How to read ELA:** Areas that are brightly white or neon indicate parts of the image saved at a different compression level. If a single person or face glows much brighter than the rest of the uniform background, it may be spliced or AI-generated.")
        
    st.markdown("---")
    st.header("Deep Learning Pipeline")
    
    # Analysis row 2: AI Models (ResNet & SigLIP)
    col3, col4 = st.columns(2)
    
    with col3:
        with st.spinner("Engine 3: Analyzing with ResNet..."):
            resnet_score = run_resnet(rgb_img)
        st.metric(label="ResNet Pipeline ('Fake' Confidence)", value=f"{resnet_score * 100:.2f}%")
        
    with col4:
        with st.spinner("Engine 4: Analyzing with SigLIP..."):
            siglip_score = run_siglip(rgb_img)
        st.metric(label="SigLIP Pipeline ('Fake' Confidence)", value=f"{siglip_score * 100:.2f}%")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Analysis row 3: Additional Advanced Models
    col5, col6 = st.columns(2)
    
    with col5:
        with st.spinner("Engine 5: Analyzing with SDXL Detector..."):
            sdxl_score = run_sdxl_detector(rgb_img)
        st.metric(label="SDXL Detector ('Fake' Confidence)", value=f"{sdxl_score * 100:.2f}%")
        
    with col6:
        with st.spinner("Engine 6: Analyzing with ViT Deepfake Detector..."):
            deepfake_score = run_deepfake_detector(rgb_img)
        st.metric(label="ViT Deepfake Detector ('Fake' Confidence)", value=f"{deepfake_score * 100:.2f}%")
        
    with final_judge_placeholder.container():
        verdict, confidence = calculate_final_verdict(resnet_score, siglip_score, sdxl_score, deepfake_score)
        
        if verdict == "FAKE":
            st.error(f"### Final Verdict: {verdict}")
            st.error(f"{confidence:.2f}% certainty that this is AI generated.")
        else:
            st.success(f"### Final Verdict: {verdict}")
            st.success(f"{confidence:.2f}% certainty that this is inherently real.")
