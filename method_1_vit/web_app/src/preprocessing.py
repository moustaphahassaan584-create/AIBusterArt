import io
import numpy as np
from PIL import Image

def prepare_image(image_file):
    """
    Accepts a Pillow image or file-like object and returns variations for downstream tasks.
    Validates image format and strips metadata.
    """
    # Load image and strip metadata by recreating it
    img = Image.open(image_file).convert('RGB')
    
    # Recreate image to strip EXIF/metadata
    data = list(img.getdata())
    clean_img = Image.new(img.mode, img.size)
    clean_img.putdata(data)
    
    # 1. Grayscale numpy array for FFT
    grayscale_img = clean_img.convert('L')
    grayscale_array = np.array(grayscale_img)
    
    # 2. 90% quality JPEG compression in memory for ELA
    buffer = io.BytesIO()
    clean_img.save(buffer, format='JPEG', quality=90)
    buffer.seek(0)
    ela_jpeg_img = Image.open(buffer).convert('RGB')
    
    # 3. Standard RGB image
    rgb_img = clean_img
    
    return grayscale_array, ela_jpeg_img, rgb_img
