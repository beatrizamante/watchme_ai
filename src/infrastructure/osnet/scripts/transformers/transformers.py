from torchvision import transforms
import torch
import numpy as np
from PIL import Image

def create_transforms(img_height, img_width):
    """Create preprocessing transforms for input images."""
    return transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image, transform):
    """
    Preprocess a single image for encoding.

    Args:
        image: PIL Image or numpy array or tensor

    Returns:
        tensor: Preprocessed image tensor
    """
    if isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            image = Image.fromarray(image)
        else:
            image = Image.fromarray((image * 255).astype(np.uint8))
    elif isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.dim() == 3 and image.shape[0] in [1, 3]:
            image = transforms.ToPILImage()(image)
        else:
            raise ValueError("Unsupported tensor format")
    elif not isinstance(image, Image.Image):
        raise ValueError("Unsupported image format. Use PIL Image, numpy array, or tensor.")

    if image.mode != 'RGB':
        image = image.convert('RGB')

    tensor = transform(image)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Transform did not return a tensor, got {type(tensor)}")

    return tensor.unsqueeze(0)
