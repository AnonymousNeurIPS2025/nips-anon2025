from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from dvgt_segmentor import DVGTSegmentation
import numpy as np
from matplotlib.colors import ListedColormap

img = Image.open('images/horses.jpg')

# Resize image to prevent INT_MAX tensor size issue
# Follow the same resizing strategy as in eval.py config
short_side = 480
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

name_list = ['sky', 'hill', 'tree', 'horse', 'grass']

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

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].axis('off')
ax[1].imshow(seg_pred, cmap='viridis')
ax[1].axis('off')
plt.tight_layout()
# plt.show()
plt.savefig('images/seg_pred.png', bbox_inches='tight')

