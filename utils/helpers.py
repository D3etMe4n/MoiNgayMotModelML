# utils/helpers.py

import base64
from io import BytesIO
import numpy as np
from PIL import Image
import torch

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def tensor_to_base64(tensor_image):
    """Convert normalized tensor to base64 encoded PNG"""
    denorm = tensor_image * 0.5 + 0.5  # Undo normalization
    image_np = denorm.numpy().transpose(1, 2, 0)  # CHW -> HWC
    image_np = (image_np * 255).astype(np.uint8)
    pil_img = Image.fromarray(image_np)
    
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")