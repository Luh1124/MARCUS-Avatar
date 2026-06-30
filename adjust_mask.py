#!/usr/bin/env python3
"""
Mask adjustment script
Adjust mask size with erosion and dilation operations.

Usage:
    # Erode the mask (shrink)
    python adjust_mask.py input_mask.png output_mask.png --erode --iterations 2 --kernel-size 5
    
    # Dilate the mask (expand)
    python adjust_mask.py input_mask.png output_mask.png --dilate --iterations 2 --kernel-size 5
    
    # Erode then dilate (opening) to remove small noise
    python adjust_mask.py input_mask.png output_mask.png --open --iterations 1 --kernel-size 3
    
    # Dilate then erode (closing) to fill small holes
    python adjust_mask.py input_mask.png output_mask.png --close --iterations 1 --kernel-size 3
    
    # Combined operation: erode then dilate, often used to shrink masks
    python adjust_mask.py input_mask.png output_mask.png --erode --iterations 2 --kernel-size 10 --then-dilate --dilate-iterations 1 --dilate-kernel-size 5
"""

import cv2
import numpy as np
from PIL import Image
import argparse
import os


def load_mask(mask_path):
    """Load a mask image"""
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {mask_path}")
    return img


def save_mask(mask, output_path):
    """Save a mask image"""
    cv2.imwrite(output_path, mask)
    print(f"[INFO] Mask saved to: {output_path}")


def erode_mask(mask, iterations=1, kernel_size=5):
    """Erode the mask (shrink)"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=iterations)
    return eroded


def dilate_mask(mask, iterations=1, kernel_size=5):
    """Dilate the mask (expand)"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated


def morph_open(mask, iterations=1, kernel_size=5):
    """Opening: erode then dilate to remove small noise"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opened


def morph_close(mask, iterations=1, kernel_size=5):
    """Closing: dilate then erode to fill small holes"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed


def get_mask_stats(mask, name="Mask"):
    """Get mask statistics"""
    total_pixels = mask.shape[0] * mask.shape[1]
    white_pixels = np.sum(mask > 0)
    coverage = (white_pixels / total_pixels) * 100
    print(f"[INFO] {name} statistics:")
    print(f"  - Size: {mask.shape[1]}x{mask.shape[0]}")
    print(f"  - White pixels: {white_pixels:,} ({coverage:.2f}%)")
    print(f"  - Black pixels: {total_pixels - white_pixels:,} ({100 - coverage:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Adjust mask image size (erosion/dilation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Erode the mask (shrink)
  python adjust_mask.py mask.png output.png --erode --iterations 2 --kernel-size 10
  
  # Dilate the mask (expand)
  python adjust_mask.py mask.png output.png --dilate --iterations 2 --kernel-size 10
  
  # Erode then dilate (opening)
  python adjust_mask.py mask.png output.png --open --iterations 1 --kernel-size 3
  
  # Dilate then erode (closing)
  python adjust_mask.py mask.png output.png --close --iterations 1 --kernel-size 3
        """
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Input mask image path"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output mask image path"
    )
    
    # Operation type (mutually exclusive)
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument(
        "--erode",
        action="store_true",
        help="Erode the mask (shrink)"
    )
    operation_group.add_argument(
        "--dilate",
        action="store_true",
        help="Dilate the mask (expand)"
    )
    operation_group.add_argument(
        "--open",
        action="store_true",
        help="Opening: erode then dilate to remove small noise"
    )
    operation_group.add_argument(
        "--close",
        action="store_true",
        help="Closing: dilate then erode to fill small holes"
    )
    
    # Erosion parameters
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Erosion/dilation iteration count (default: 1)"
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=5,
        help="Erode/Dilate kernel size (default: 5)"
    )
    
    # Combined operation: erode then dilate
    parser.add_argument(
        "--then-dilate",
        action="store_true",
        help="Run dilation after erosion to smooth edges"
    )
    parser.add_argument(
        "--dilate-iterations",
        type=int,
        default=1,
        help="Dilation iteration count (default: 1)"
    )
    parser.add_argument(
        "--dilate-kernel-size",
        type=int,
        default=5,
        help="Dilation kernel size (default: 5)"
    )
    
    # Visualization options
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show mask statistics"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview the result (requires matplotlib)"
    )
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file does not exist: {args.input}")
        return 1
    
    # Load original mask
    print(f"[INFO] Loading mask: {args.input}")
    original_mask = load_mask(args.input)
    
    if args.show_stats:
        get_mask_stats(original_mask, "Original Mask")
    
    # Run operation
    result_mask = original_mask.copy()
    
    if args.erode:
        print(f"[INFO] Running erosion (iterations={args.iterations}, kernel_size={args.kernel_size})")
        result_mask = erode_mask(result_mask, args.iterations, args.kernel_size)
        
        if args.then_dilate:
            print(f"[INFO] Running follow-up dilation (iterations={args.dilate_iterations}, kernel_size={args.dilate_kernel_size})")
            result_mask = dilate_mask(result_mask, args.dilate_iterations, args.dilate_kernel_size)
    
    elif args.dilate:
        print(f"[INFO] Running dilation (iterations={args.iterations}, kernel_size={args.kernel_size})")
        result_mask = dilate_mask(result_mask, args.iterations, args.kernel_size)
    
    elif args.open:
        print(f"[INFO] Running opening (iterations={args.iterations}, kernel_size={args.kernel_size})")
        result_mask = morph_open(result_mask, args.iterations, args.kernel_size)
    
    elif args.close:
        print(f"[INFO] Running closing (iterations={args.iterations}, kernel_size={args.kernel_size})")
        result_mask = morph_close(result_mask, args.iterations, args.kernel_size)
    
    if args.show_stats:
        get_mask_stats(result_mask, "Processed Mask")
        
        # Compute change
        original_white = np.sum(original_mask > 0)
        result_white = np.sum(result_mask > 0)
        change = result_white - original_white
        change_percent = (change / original_white) * 100 if original_white > 0 else 0
        print(f"[INFO] Change:")
        print(f"  - White pixelsChange: {change:+,} ({change_percent:+.2f}%)")
    
    # Save result
    save_mask(result_mask, args.output)
    
    # Preview (optional)
    if args.preview:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(original_mask, cmap='gray')
            axes[0].set_title('Original Mask')
            axes[0].axis('off')
            axes[1].imshow(result_mask, cmap='gray')
            axes[1].set_title('Processed Mask')
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("[WARNING] matplotlib is not installed; skipping preview")
    
    print("[INFO] Done!")
    return 0


if __name__ == "__main__":
    exit(main())
