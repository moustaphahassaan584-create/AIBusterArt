import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from transformers import pipeline
import streamlit as st

# We use st.cache_resource to load pipelines only once and prevent Streamlit from reloading them on every interaction.
@st.cache_resource
def load_resnet_pipeline():
    return pipeline("image-classification", model="umm-maybe/AI-image-detector")

@st.cache_resource
def load_siglip_pipeline():
    return pipeline("image-classification", model="Ateeqq/ai-vs-human-image-detector")

@st.cache_resource
def load_sdxl_pipeline():
    return pipeline("image-classification", model="Organika/sdxl-detector")

@st.cache_resource
def load_deepfake_pipeline():
    return pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model")

def run_fft(grayscale_array):
    """
    Calculates the Fast Fourier Transform magnitude spectrum on the grayscale array.
    Returns a matplotlib figure.
    """
    f = np.fft.fft2(grayscale_array)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(magnitude_spectrum, cmap='gray')
    ax.axis('off')
    ax.set_title('FFT Magnitude Spectrum')
    plt.tight_layout()
    
    return fig

def run_ela(original_img, jpeg_img):
    """
    Computes Error Level Analysis difference using the original and 90% quality JPEG.
    Returns the enhanced ELA Pillow image.
    """
    # ELA works by diffing the original against a re-compressed version
    diff = ImageChops.difference(original_img, jpeg_img)
    
    # Many ELA implementations use a constant multiplier instead of dividing by
    # max_diff, because a single bright pixel of noise can cause max_diff to push
    # the entire image into darkness. A multiplier of 15 perfectly enhances the
    # subtle block artifacts without washing everything out.
    ela_img = Image.eval(diff, lambda x: min(255, x * 15.0))
    
    return ela_img

def run_resnet(image):
    """
    Uses the umm-maybe/AI-image-detector pipeline.
    Returns the float confidence score for 'fake'/'AI'.
    """
    resnet_pipeline = load_resnet_pipeline()
    results = resnet_pipeline(image)
    
    fake_score = 0.0
    for res in results:
        # Check label name specific to the umm-maybe model (typically 'artificial' or 'fake')
        label_lower = res['label'].lower()
        if label_lower in ['artificial', 'fake', 'ai']:
            fake_score = res['score']
            break
        elif label_lower in ['human', 'real']:
            fake_score = 1.0 - res['score']
            
    return float(fake_score)

def run_siglip(image):
    """
    Uses the Ateeqq/ai-vs-human-image-detector pipeline.
    Returns the float confidence score for 'fake'/'AI'.
    """
    siglip_pipeline = load_siglip_pipeline()
    results = siglip_pipeline(image)
    
    fake_score = 0.0
    for res in results:
        # Check label name specific to the Ateeq model (typically 'artificial' or 'AI Generated')
        label_lower = res['label'].lower()
        if label_lower in ['artificial', 'fake', 'ai', 'ai generated']:
            fake_score = res['score']
            break
        elif label_lower in ['human', 'real']:
            fake_score = 1.0 - res['score']
            
    return float(fake_score)

def run_sdxl_detector(image):
    """
    Uses the Organika/sdxl-detector pipeline.
    Returns the float confidence score for 'fake'/'AI'.
    """
    sdxl_pipeline = load_sdxl_pipeline()
    results = sdxl_pipeline(image)
    
    fake_score = 0.0
    for res in results:
        label_lower = res['label'].lower()
        if label_lower in ['artificial', 'fake', 'ai', 'ai generated']:
            fake_score = res['score']
            break
        elif label_lower in ['human', 'real']:
            fake_score = 1.0 - res['score']
            
    return float(fake_score)

def run_deepfake_detector(image):
    """
    Uses the prithivMLmods/Deep-Fake-Detector-v2-Model pipeline.
    Returns the float confidence score for 'fake'/'Deepfake'.
    """
    deepfake_pipeline = load_deepfake_pipeline()
    results = deepfake_pipeline(image)
    
    fake_score = 0.0
    for res in results:
        label_lower = res['label'].lower()
        if label_lower in ['deepfake', 'fake', 'artificial', 'ai']:
            fake_score = res['score']
            break
        elif label_lower in ['realism', 'real', 'human']:
            fake_score = 1.0 - res['score']
            
    return float(fake_score)
