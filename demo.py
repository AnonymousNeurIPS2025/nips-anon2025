from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from dvgt_segmentor import DVGTSegmentation
import numpy as np
from matplotlib.colors import ListedColormap
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='DVGT Segmentation Demo')
    parser.add_argument('--image', default='images/horses.jpg', 
                        help='Path to input image (default: images/horses.jpg)')
    parser.add_argument('--classes', nargs='+', default=['sky', 'hill', 'tree', 'horse', 'grass'],
                        help='List of class names (default: sky hill tree horse grass)')
    parser.add_argument('--output', default='images/seg_pred.png',
                        help='Path to save output image (default: images/seg_pred.png)')
    parser.add_argument('--short-side', type=int, default=480,
                        help='Short side length for resizing (default: 480)')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Alpha value for overlay transparency (default: 0.6)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if input image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        return
    
    img = Image.open(args.image)
    print(f"Loaded image: {args.image}")
    print(f"Classes: {args.classes}")

    # Resize image to prevent INT_MAX tensor size issue
    # Follow the same resizing strategy as in eval.py config
    short_side = args.short_side
    w, h = img.size
    if w < h:
        new_w = short_side
        new_h = int(h * short_side / w)
    else:
        new_h = short_side  
        new_w = int(w * short_side / h)

    # Ensure max dimension doesn't exceed 2048 (as in config)
    max_size = 2048
    if max(new_w, new_h) > max_size:
        if new_w > new_h:
            new_h = int(new_h * max_size / new_w)
            new_w = max_size
        else:
            new_w = int(new_w * max_size / new_h)
            new_h = max_size

    img = img.resize((new_w, new_h), Image.LANCZOS)
    print(f"Resized image to: {img.size}")

    name_list = args.classes

    with open('./configs/my_name.txt', 'w') as writers:
        for i in range(len(name_list)):
            if i == len(name_list) - 1:
                writers.write(name_list[i])
            else:
                writers.write(name_list[i] + '\n')
        writers.close()

    img_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])(img)

    img_tensor = img_tensor.unsqueeze(0).to('cuda')

    model = DVGTSegmentation(clip_type='openai', 
                             model_type='convnext_l', 
                             name_path='./configs/my_name.txt')

    seg_pred = model.predict(img_tensor, data_samples=None)
    seg_pred = seg_pred.data.cpu().numpy().squeeze(0)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Original image
    ax[0].imshow(img)
    ax[0].set_title('Original Image', fontsize=14)
    ax[0].axis('off')

    # Create overlay of original image and segmentation
    overlay = img.copy()
    overlay_array = np.array(overlay)

    # Create colored mask for each class
    alpha = args.alpha
    cmap_tab10 = plt.colormaps.get_cmap('tab10')
    for class_idx in range(len(name_list)):
        mask = (seg_pred == class_idx)
        if np.any(mask):
            color = np.array(cmap_tab10(class_idx)[:3]) * 255
            overlay_array[mask] = overlay_array[mask] * (1 - alpha) + color * alpha

    ax[1].imshow(overlay_array.astype(np.uint8))
    ax[1].set_title('Segmentation Result', fontsize=14)
    ax[1].axis('off')

    # Add legend for overlay
    from matplotlib.patches import Patch
    unique_classes = np.unique(seg_pred)
    legend_elements = [Patch(facecolor=cmap_tab10(i), label=name_list[i]) 
                      for i in range(len(name_list)) if i in unique_classes]
    if legend_elements:
        ax[1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(args.output, bbox_inches='tight', dpi=150)
    print(f"Segmentation result saved to: {args.output}")

if __name__ == '__main__':
    main()

