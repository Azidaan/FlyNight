# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from tqdm import tqdm
import torch
from torchvision.utils import draw_bounding_boxes
from PIL import Image
from PIL import ImageOps
import torchvision
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
from torchvision.io import decode_image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

'''
# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
'''

@torch.no_grad()
def get_mean_std(loader):
    """
    Compute dataset mean and std (per-channel) over a dataloader.
    Now includes tqdm progress bar for feedback.
    """
    channel_sum = None
    channel_sq_sum = None
    n_pixels = 0

    # tqdm progress bar
    for data, *_ in tqdm(loader, desc="Computing mean/std", unit="batch"):
        if not torch.is_floating_point(data):
            data = data.float()

        # scale to [0, 1] if values are 0–255
        if data.max() > 1.0:
            data = data / 255.0

        b, c, h, w = data.shape
        pixels = b * h * w

        s  = data.sum(dim=(0, 2, 3))          # per-channel sum
        s2 = (data ** 2).sum(dim=(0, 2, 3))   # per-channel squared sum

        if channel_sum is None:
            channel_sum = s.double()
            channel_sq_sum = s2.double()
        else:
            channel_sum += s.double()
            channel_sq_sum += s2.double()

        n_pixels += pixels

    mean = channel_sum / n_pixels
    std = torch.sqrt(channel_sq_sum / n_pixels - mean ** 2)

    print(f"\n[INFO] mean = {mean.tolist()}")
    print(f"[INFO] std  = {std.tolist()}")

    return mean.float(), std.float()


# Fast option 2: torchvision (returns tensor, convert to numpy if needed)
def load_image_torchvision(path):
    with open(path, 'rb') as f:
        image_bytes = f.read()
    image = decode_image(torch.frombuffer(image_bytes, dtype=torch.uint8))  # Returns tensor in RGB
    return image.permute(1, 2, 0).numpy()  # Convert to numpy if needed

def id2name_img_coco(img_id):
    # 000000055022
    return f"{img_id:012d}"

def save_img_cv2(path, img):
    cv2.imwrite(path, img)

def crop_bbox_224(fimg, labels, bboxes):
    transform = A.Compose([
        A.Resize(height=255, width=255),
        A.CenterCrop(height=224, width=224),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    image = cv2.imread(fimg)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # visualize_bbox_plt(image, bboxes, labels)
    transformed = transform(image=image, bboxes=bboxes, class_labels=labels)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_class_labels = transformed['class_labels']
    #visualize_bbox_plt(transformed_image, transformed_bboxes, transformed_class_labels)
    return transformed_image, transformed_bboxes, transformed_class_labels



def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def convert_bboxes():
    pass

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def crop_by_bbox_with_fill(h=224, w=224, img_path='', bbox=[], color='black', fill=False):
    # pascal-voc format
    x_min, y_min, x_max, y_max = bbox
    img = Image.open(img_path)
    img = img.convert('RGB')
    # width, height = img.size
    # width = x_max - x_min
    # height = y_max - y_min
    img = img.crop([x_min, y_min, x_max, y_max])
    if fill:
        try:
            img = ImageOps.pad(img, (h, w), color=color)
        except:
            img = img.resize((h, w))
    return img

def draw_and_show_bbox_tv(img_path='', bbox='', flag_show=True):
    """
    :param img_path:
    :param bbox:
    :return:
    """
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = torchvision.transforms.PILToTensor()(img)
    box = torch.tensor(bbox)
    box = box.unsqueeze(0)
    # draw bounding box and fill color
    img = draw_bounding_boxes(img, box, width=5, colors="green", fill=True)
    # transform this image to PIL image
    img = torchvision.transforms.ToPILImage()(img)
    # display output
    img.show()
    return img

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_bbox_func(img, bbox, class_name,  thickness=2):
    """Visualizes a single bounding box on the image"""
    # https://www.rapidtables.com/web/color/RGB_Color.html
    BOX_COLOR = (228, 28, 28) # Red
    TEXT_COLOR = (255, 255, 255) # White
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=BOX_COLOR, thickness=thickness)
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize_bbox_plt(image, bboxes, labels):
    img = image.copy()
    for bbox, label in zip(bboxes, labels):
        #class_name = category_id_to_name[category_id]
        img = visualize_bbox_func(img, bbox, label)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    #cv2.imwrite('1.png', img)

def tensor_from_img(path):

    # check for rgb mode
    img = Image.open(path)
    img = img.convert('RGB')
    from torchvision import transforms
    resize = transforms.Resize([224, 224])
    to_tensor = transforms.ToTensor()
    img = resize(img)
    tensor = to_tensor(img)
    tensor = tensor.unsqueeze(0)
    return tensor

if __name__ == "_main__":
    pass