import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_gamma_inverse(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """
    Apply inverse gamma to an image, commonly used to darken highlights for highlight compression.
    Args:
        img: Input image as a NumPy array with value range 0-255 or 0-1
        gamma: Gamma value, usually 2.2 or 2.4
    Returns:
        NumPy array with the same shape as input
    """
    img_float = img.astype(np.float32) / 255. if img.max() > 1.0 else img.astype(np.float32)
    out = np.power(img_float, gamma)
    out = np.clip(out * 255, 0, 255) if img.max() > 1.0 else np.clip(out, 0, 1)
    return out.astype(img.dtype)

def apply_gamma(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """
    Apply standard gamma, equivalent to lifting highlights and often used to decode sRGB; rarely used for highlight compression but provided for comparison.
    """
    img_float = img.astype(np.float32) / 255. if img.max() > 1.0 else img.astype(np.float32)
    out = np.power(img_float, 1.0/gamma)
    out = np.clip(out * 255, 0, 255) if img.max() > 1.0 else np.clip(out, 0, 1)
    return out.astype(img.dtype)

def apply_sigmoid_curve(img: np.ndarray, alpha: float = 15, beta: float = 0.75) -> np.ndarray:
    """
    Use a sigmoid curve mainly to suppress highlights while leaving dark regions mostly unchanged.
    alpha controls steepness and beta is the highlight knee point.
    Recommended: alpha=15, beta=0.75 for highlight-only compression.
    """
    img_float = img.astype(np.float32)/255. if img.max() > 1.0 else img.astype(np.float32)
    # Below beta the curve is approximately linear; above beta it drops steeply.
    out = img_float * (img_float < beta) + \
        (1/(1+np.exp(-alpha*(img_float - beta)))) * (img_float >= beta)
    # Normalize the sigmoid segment to [0, 1]
    sigmoid_mask = img_float >= beta
    if np.any(sigmoid_mask):
        s = (1/(1+np.exp(-alpha*(np.array([1.0]) - beta)))) - (1/(1+np.exp(-alpha*(beta - beta))))
        m = 1/(1+np.exp(-alpha*(beta - beta)))
        out[sigmoid_mask] = (out[sigmoid_mask] - m) / (s + 1e-6) + beta
        out[sigmoid_mask] = np.clip(out[sigmoid_mask], 0, 1)
    out = np.clip(out * 255, 0, 255) if img.max() > 1.0 else np.clip(out, 0, 1)
    return out.astype(img.dtype)

def tone_curve_highlight_compress(img: np.ndarray, highlight_start: float = 0.6, highlight_gamma: float = 3.5) -> np.ndarray:
    """
    Piecewise curve: dark and mid regions are mostly linear, highlights above highlight_start are compressed with a larger gamma power.
    highlight_start: highlight start (0-1), recommended 0.7
    highlight_gamma: highlight compression strength; values >1 increase compression, recommended 3.5.
    """
    img_float = img.astype(np.float32) / 255. if img.max() > 1.0 else img.astype(np.float32)
    out = np.zeros_like(img_float)
    mask_shadow = img_float < highlight_start
    mask_highlight = ~mask_shadow
    out[mask_shadow] = img_float[mask_shadow]
    # Use power compression for highlights
    out[mask_highlight] = highlight_start + (1 - highlight_start) * np.power(
        (img_float[mask_highlight] - highlight_start) / (1 - highlight_start), highlight_gamma
    )
    out = np.clip(out * 255, 0, 255) if img.max() > 1.0 else np.clip(out, 0, 1)
    return out.astype(img.dtype)

def plot_curves():
    """
    Visualize the curves above for highlight compression / brightness adjustment; after adjustment, dark regions stay linear and highlights are clearly compressed.
    """
    x = np.linspace(0, 1, 256)
    plt.figure(figsize=(8,6))
    plt.plot(x, np.power(x, 2.2), label='Gamma Inv (2.2)', color='r')
    plt.plot(x, np.power(x, 1/2.2), label='Gamma 2.2', color='b')

    # New sigmoid highlight curve
    y_sigmoid = x * (x < 0.75) + (1/(1+np.exp(-15*(x-0.75)))) * (x >= 0.75)
    # Normalize sigmoid segment
    s = (1/(1+np.exp(-15*(1.0-0.75)))) - (1/(1+np.exp(-15*(0.75-0.75))))
    m = 1/(1+np.exp(-15*(0.75-0.75)))
    mask = x >= 0.75
    if np.any(mask):
        y_sigmoid[mask] = (y_sigmoid[mask] - m) / (s + 1e-6) + 0.75
        y_sigmoid[mask] = np.clip(y_sigmoid[mask], 0, 1)
    plt.plot(x, y_sigmoid, label='Sigmoid-Highlight', color='g')

    # Piecewise highlight compression curve
    highlight_start = 0.7
    highlight_gamma = 3.5
    y_curve = np.zeros_like(x)
    mask2 = x < highlight_start
    mask3 = x >= highlight_start
    y_curve[mask2] = x[mask2]
    y_curve[mask3] = highlight_start + (1 - highlight_start) * np.power(
        (x[mask3] - highlight_start)/(1-highlight_start), highlight_gamma)
    plt.plot(x, y_curve, label='Tone-Highlight Compress', color='m')
    plt.legend()
    plt.title('Highlight Compression Curves (Bright Region Only)')
    plt.grid()
    plt.show()

def image_to_array(img: Image.Image) -> np.ndarray:
    """Convert PIL.Image to np.ndarray, supporting grayscale and color automatically."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    return arr

def array_to_image(arr: np.ndarray) -> Image.Image:
    """Convert ndarray to PIL.Image, supporting grayscale and color automatically."""
    arr = np.squeeze(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)

def process_image_curve(img_path, method='gamma_inv', **kwargs):
    """
    Load an image and apply the selected highlight compression / brightness adjustment method.
    method: 'gamma_inv', 'gamma', 'sigmoid', 'tone_curve'
    """
    img = Image.open(img_path).convert("RGB")
    arr = image_to_array(img)
    if method == 'gamma_inv':
        out = apply_gamma_inverse(arr, **kwargs)
    elif method == 'gamma':
        out = apply_gamma(arr, **kwargs)
    elif method == 'sigmoid':
        out = apply_sigmoid_curve(arr, **kwargs)
    elif method == 'tone_curve':
        out = tone_curve_highlight_compress(arr, **kwargs)
    else:
        raise ValueError('Unknown method: %s' % method)
    return array_to_image(out)

import os

def show_compare(original: Image.Image, processed: Image.Image, title1="Original", title2="Processed", save_dir="results", suffix=""):
    """Show original and result side by side and save locally without displaying a window."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    axs[0].imshow(original)
    axs[0].set_title(title1)
    axs[0].axis('off')
    axs[1].imshow(processed)
    axs[1].set_title(title2)
    axs[1].axis('off')
    # Generate a distinguishable filename
    out_name = f"color_transfer_compare{suffix}.png"
    out_path = os.path.join(save_dir, out_name)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Image comparison saved to: {out_path}")

if __name__ == "__main__":
    # Demo image path; replace with your own path if needed
    test_img_path = os.environ.get("MARCUS_COLOR_TEST_IMAGE", "examples/000205.jpg")
    img = Image.open(test_img_path).convert("RGB")
    # Test multiple methods
    img_gamma_inv = process_image_curve(test_img_path, method='gamma_inv', gamma=2.2)
    img_sigmoid = process_image_curve(test_img_path, method='sigmoid', alpha=15, beta=0.75)
    img_tone_curve = process_image_curve(test_img_path, method='tone_curve', highlight_start=0.7, highlight_gamma=3.5)

    print('Save comparison images for highlight compression and brightening methods...')
    show_compare(img, img_gamma_inv, "Original", "Inverse gamma (darken highlights)", suffix="_gamma_inv")
    show_compare(img, img_sigmoid, "Original", "Custom sigmoid highlight curve", suffix="_sigmoid")
    show_compare(img, img_tone_curve, "Original", "Piecewise highlight compression", suffix="_tone_curve")
    print('Save reference curves...')
    if not os.path.exists("results"):
        os.makedirs("results")
    plot_curves()
    plt.savefig(os.path.join("results", "curve_reference.png"), bbox_inches="tight")
    plt.close()
    print('Curve plot saved to: results/curve_reference.png')
