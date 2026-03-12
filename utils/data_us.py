from random import randint
import numpy as np
import torch
from skimage import io, color, measure  # Add measure for connected component analysis
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import cv2
import pandas as pd
from collections import defaultdict
from batchgenerators.utilities.file_and_folder_operations import *
from torchvision.transforms import InterpolationMode
import json  # Add json import


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


def get_connected_components(mask, class_id=1):
    """Get connected components and return region list"""
    binary_mask = (mask == class_id).astype(np.uint8)
    if np.sum(binary_mask) == 0:
        return []  # No target regions

    labeled_mask, num_labels = measure.label(binary_mask, connectivity=2, return_num=True)
    regions = []
    for label in range(1, num_labels + 1):
        region_mask = (labeled_mask == label)
        regions.append(region_mask)
    return regions


def random_click(mask, class_id=1, max_prompts=10):
    """Support random point generation for multiple mask regions"""
    regions = get_connected_components(mask, class_id)
    point_label = 1

    if not regions:  # No target regions
        indices = np.argwhere(mask != class_id)
        if len(indices) == 0:
            # Entire image is target? Impossible, but just in case
            indices = np.argwhere(np.ones_like(mask))
        indices = indices[:, [1, 0]]  # Convert to (x,y)
        pt_idx = np.random.randint(len(indices), size=1)
        pt = indices[pt_idx]
        return pt, [0]  # Background point label is 0

    # Randomly select regions (up to max_prompts)
    if len(regions) > max_prompts:
        selected_indices = np.random.choice(len(regions), max_prompts, replace=False)
        regions = [regions[i] for i in selected_indices]

    points = []
    labels = []
    for region in regions:
        indices = np.argwhere(region)
        indices = indices[:, [1, 0]]  # Convert to (x,y)
        pt_idx = np.random.randint(len(indices))
        pt = indices[pt_idx]
        points.append(pt)
        labels.append(point_label)

    return np.array(points), labels


def fixed_click(mask, class_id=1, max_prompts=10):
    """Support fixed point generation for multiple mask regions (using region center points)"""
    regions = get_connected_components(mask, class_id)
    point_label = 1

    if not regions:  # No target regions
        h, w = mask.shape
        center = [w // 2, h // 2]  # Image center
        return np.array([center]), [0]  # Background point label is 0

    # Select regions (up to max_prompts)
    if len(regions) > max_prompts:
        # Sort by area, select top max_prompts largest regions
        region_areas = [np.sum(region) for region in regions]
        sorted_indices = np.argsort(region_areas)[::-1][:max_prompts]
        regions = [regions[i] for i in sorted_indices]

    points = []
    labels = []
    for region in regions:
        # Calculate region center
        indices = np.argwhere(region)
        center_y, center_x = np.mean(indices, axis=0)
        pt = [int(center_x), int(center_y)]  # Convert to (x,y)
        points.append(pt)
        labels.append(point_label)

    return np.array(points), labels


def random_clicks(mask, class_id=1, prompts_number=10):
    """This function remains unchanged"""
    indices = np.argwhere(mask == class_id)
    indices[:, [0, 1]] = indices[:, [1, 0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0, 1]] = indices[:, [1, 0]]
    pt_index = np.random.randint(len(indices), size=prompts_number)
    pt = indices[pt_index]
    point_label = np.repeat(point_label, prompts_number)
    return pt, point_label


def pos_neg_clicks(mask, class_id=1, pos_prompt_number=5, neg_prompt_number=5):
    """This function remains unchanged"""
    pos_indices = np.argwhere(mask == class_id)
    pos_indices[:, [0, 1]] = pos_indices[:, [1, 0]]
    pos_prompt_indices = np.random.randint(len(pos_indices), size=pos_prompt_number)
    pos_prompt = pos_indices[pos_prompt_indices]
    pos_label = np.repeat(1, pos_prompt_number)

    neg_indices = np.argwhere(mask != class_id)
    neg_indices[:, [0, 1]] = neg_indices[:, [1, 0]]
    neg_prompt_indices = np.random.randint(len(neg_indices), size=neg_prompt_number)
    neg_prompt = neg_indices[neg_prompt_indices]
    neg_label = np.repeat(0, neg_prompt_number)

    pt = np.vstack((pos_prompt, neg_prompt))
    point_label = np.hstack((pos_label, neg_label))
    return pt, point_label


def random_bbox(mask, class_id=1, img_size=256, max_boxes=10):
    """Support random box generation for multiple mask regions"""
    regions = get_connected_components(mask, class_id)

    if not regions:  # No target regions
        return np.array([[-1, -1, img_size, img_size]])

    # Randomly select regions (up to max_boxes)
    if len(regions) > max_boxes:
        selected_indices = np.random.choice(len(regions), max_boxes, replace=False)
        regions = [regions[i] for i in selected_indices]

    boxes = []
    for region in regions:
        indices = np.argwhere(region)  # Y X
        if len(indices) == 0:
            continue

        min_y, min_x = np.min(indices, axis=0)
        max_y, max_x = np.max(indices, axis=0)

        # Convert to X Y coordinates
        min_x, min_y, max_x, max_y = min_x, min_y, max_x, max_y

        classw_size = max_x - min_x + 1
        classh_size = max_y - min_y + 1

        shiftw = randint(int(0.95 * classw_size), int(1.05 * classw_size))
        shifth = randint(int(0.95 * classh_size), int(1.05 * classh_size))
        shiftx = randint(-int(0.05 * classw_size), int(0.05 * classw_size))
        shifty = randint(-int(0.05 * classh_size), int(0.05 * classh_size))

        new_centerx = (min_x + max_x) // 2 + shiftx
        new_centery = (min_y + max_y) // 2 + shifty

        minx = np.max([new_centerx - shiftw // 2, 0])
        maxx = np.min([new_centerx + shiftw // 2, img_size - 1])
        miny = np.max([new_centery - shifth // 2, 0])
        maxy = np.min([new_centery + shifth // 2, img_size - 1])

        boxes.append([minx, miny, maxx, maxy])

    if not boxes:  # No boxes generated
        return np.array([[-1, -1, img_size, img_size]])

    return np.array(boxes)


def fixed_bbox(mask, class_id=1, img_size=256, max_boxes=10):
    """Support fixed box generation for multiple mask regions (using minimum bounding rectangles)"""
    regions = get_connected_components(mask, class_id)

    if not regions:  # No target regions
        return np.array([[-1, -1, img_size, img_size]])

    # Sort by area, select top max_boxes largest regions
    if len(regions) > max_boxes:
        region_areas = [np.sum(region) for region in regions]
        sorted_indices = np.argsort(region_areas)[::-1][:max_boxes]
        regions = [regions[i] for i in sorted_indices]

    boxes = []
    for region in regions:
        indices = np.argwhere(region)  # Y X
        if len(indices) == 0:
            continue

        min_y, min_x = np.min(indices, axis=0)
        max_y, max_x = np.max(indices, axis=0)

        # Convert to X Y coordinates
        min_x, min_y, max_x, max_y = min_x, min_y, max_x, max_y
        boxes.append([min_x, min_y, max_x, max_y])

    if not boxes:  # No boxes generated
        return np.array([[-1, -1, img_size, img_size]])

    return np.array(boxes)

class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separately.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """

    def __init__(self, img_size=256, low_img_size=256, ori_size=256, crop=(32, 32), p_flip=0.0, p_rota=0.0, p_scale=0.0, p_gaussn=0.0, p_contr=0.0,
                 p_gama=0.0, p_distor=0.0, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_random_affine=0,
                 long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask
        self.low_img_size = low_img_size
        self.ori_size = ori_size

    def __call__(self, image, mask):
        #  gamma enhancement
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)
        # converting to uint8 if necessary
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        # transforming to PIL image

        # print(f"The data type of image: {image.dtype}")  # e.g., uint8
        # print(f"The data type of mask: {mask.dtype}")  # e.g., int64 or uint8
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random horizontal flip
        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)
        # random rotation
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)
        # random scale and center resize to the original size
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_h, new_w = int(self.img_size * scale), int(self.img_size * scale)
            image, mask = F.resize(image, (new_h, new_w), InterpolationMode.BILINEAR), F.resize(mask, (new_h, new_w), InterpolationMode.NEAREST)
            # image = F.center_crop(image, (self.img_size, self.img_size))
            # mask = F.center_crop(mask, (self.img_size, self.img_size))
            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random add gaussian noise
        if np.random.rand() < self.p_gaussn:
            ns = np.random.randint(3, 15)
            noise = np.random.normal(loc=0, scale=1, size=(self.img_size, self.img_size)) * ns
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = F.to_pil_image(image.astype('uint8'))
        # random change the contrast
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image = contr_tf(image)
        # random distortion
        if np.random.rand() < self.p_distortion:
            distortion = T.RandomAffine(0, None, None, (5, 30))
            image = distortion(image)
        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)
        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)
        # transforming to tensor
        image, mask = F.resize(image, (self.img_size, self.img_size), InterpolationMode.BILINEAR), F.resize(mask, (self.ori_size, self.ori_size), InterpolationMode.NEAREST)
        low_mask = F.resize(mask, (self.low_img_size, self.low_img_size), InterpolationMode.NEAREST)
        image = F.to_tensor(image)

        if not self.long_mask:
            mask = F.to_tensor(mask)
            low_mask = F.to_tensor(low_mask)
        else:
            mask = to_long_tensor(mask)
            low_mask = to_long_tensor(low_mask)
        return image, mask, low_mask


class ImageToImage2D(Dataset):
    def __init__(self, dataset_path: str, split_path: str, split='train', joint_transform: Callable = None, img_size=256, use_coarse_mask=True,
                 preseg_path=None, gen_name=None,
                 prompt="click", class_id=1, one_hot_mask: int = False, max_prompts=1) -> None:
        self.preseg_path = preseg_path
        self.gen_name = gen_name
        self.dataset_path = dataset_path
        self.one_hot_mask = one_hot_mask
        self.split = split
        id_list_file = os.path.join(split_path, '{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.prompt = prompt
        self.img_size = img_size
        self.class_id = class_id
        self.class_dict_file = os.path.join(dataset_path, 'MainPatient/class.json')
        with open(self.class_dict_file, 'r') as load_f:
            self.class_dict = json.load(load_f)
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))
        self.max_prompts = max_prompts  # Maximum number of prompt points
        self.max_boxes = 1
        self.use_coarse_mask = use_coarse_mask
        print("{} dataset txt path: {}".format(split, id_list_file))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        if "test" in self.split:
            class_id0, sub_path, filename = id_.split('/')[0], id_.split('/')[1], id_.split('/')[2]
        else:
            class_id0, sub_path, filename = id_.split('/')[0], id_.split('/')[1], id_.split('/')[2]

        if sub_path == "BUSI" or sub_path == "UDIAT":
            sub_path = "Breast-" + sub_path
        elif sub_path == "TN3K" or sub_path == "DDTI":
            sub_path = "ThyroidNodule-" + sub_path
        img_path = os.path.join(os.path.join(self.dataset_path, sub_path), 'img')
        label_path = os.path.join(os.path.join(self.dataset_path, sub_path), 'label')
        if self.use_coarse_mask:
            coarse_mask_path = os.path.join(os.path.join(self.preseg_path), 'coarse_mask')
        else:
            coarse_mask_path = os.path.join(os.path.join(self.preseg_path), 'fine_mask')

        generated_path = os.path.join(os.path.join(self.preseg_path, "SD15_infer"), self.gen_name, "generated")
        original_image = cv2.imread(os.path.join(img_path, filename + '.png'), 0)
        mask = cv2.imread(os.path.join(label_path, filename + '.png'), 0)
        coarse_mask = cv2.imread(os.path.join(coarse_mask_path, filename + '.png'), 0)
        generated_image = cv2.imread(os.path.join(generated_path, filename + '.png'), 0)

        classes = self.class_dict[sub_path]
        if classes == 2:
            mask[mask > 1] = 1
            coarse_mask[coarse_mask > 1] = 1

        original_image, mask = correct_dims(original_image, mask)
        _, coarse_mask = correct_dims(original_image, coarse_mask)
        generated_image = correct_dims(generated_image)

        if self.joint_transform:
            origin_mask = mask
            source_image = original_image
            original_image, mask, low_mask = self.joint_transform(original_image, mask)
            _, coarse_mask, low_coarse_mask = self.joint_transform(source_image, coarse_mask)
            generated_image, _, _ = self.joint_transform(generated_image, origin_mask)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
            coarse_mask = torch.zeros((self.one_hot_mask, coarse_mask.shape[1], coarse_mask.shape[2])).scatter_(0, coarse_mask.long(), 1)

        if self.prompt == 'click':
            point_label = 1
            if 'train' in self.split:
                class_id = int(class_id0)
            elif 'val' in self.split:
                class_id = int(class_id0)
            else:
                class_id = self.class_id

            if 'train' in self.split:
                pt, point_label = random_click(np.array(mask), class_id, max_prompts=self.max_prompts)
                coarse_bbox = random_bbox(np.array(mask), class_id, self.img_size, max_boxes=1)
            else:
                pt, point_label = fixed_click(np.array(mask), class_id, max_prompts=self.max_prompts)
                coarse_bbox = fixed_bbox(np.array(mask), class_id, self.img_size, max_boxes=1)

            mask[mask!=class_id] = 0
            coarse_mask[coarse_mask!=class_id] = 0
            mask[mask==class_id] = 1
            coarse_mask[coarse_mask==class_id] = 1
            low_mask[low_mask!=class_id] = 0
            low_coarse_mask[low_coarse_mask!=class_id] = 0
            low_mask[low_mask==class_id] = 1
            low_coarse_mask[low_coarse_mask==class_id] = 1

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
            coarse_mask = torch.zeros((self.one_hot_mask, coarse_mask.shape[1], coarse_mask.shape[2])).scatter_(0, coarse_mask.long(), 1)

        low_mask = low_mask.unsqueeze(0)
        low_coarse_mask = low_coarse_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)
        coarse_mask = coarse_mask.unsqueeze(0)

        # Pad point_labels
        point_label_padded = np.zeros(self.max_prompts)
        point_label_padded[:len(point_label)] = point_label

        # Pad pt
        pt_padded = np.zeros((self.max_prompts, 2))
        pt_padded[:pt.shape[0]] = pt

        # Pad bbox
        bbox_padded = np.zeros((self.max_boxes, 4))
        bbox_padded[:coarse_bbox.shape[0]] = coarse_bbox


        return {
            'image': original_image,
            'generated_image': generated_image,
            'label': mask,
            'p_label': point_label_padded,
            'pt': pt_padded,
            'bbox': bbox_padded,
            'low_mask': low_mask,
            'low_coarse_mask': low_coarse_mask,
            'coarse_mask': coarse_mask,
            'image_name': filename + '.png',
            'class_id': class_id
        }

class PreSegImageToImage2D(Dataset):
    def __init__(self, dataset_path: str, split_path: str, split='train', joint_transform: Callable = None, img_size=256, class_id=1) -> None:
        self.dataset_path = dataset_path
        self.split = split
        id_list_file = os.path.join(split_path, '{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.img_size = img_size
        self.class_id = class_id
        self.class_dict_file = os.path.join(dataset_path, 'MainPatient/class.json')
        with open(self.class_dict_file, 'r') as load_f:
            self.class_dict = json.load(load_f)
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        if "test" in self.split:
            class_id0, sub_path, filename = id_.split('/')[0], id_.split('/')[1], id_.split('/')[2]
        else:
            class_id0, sub_path, filename = id_.split('/')[0], id_.split('/')[1], id_.split('/')[2]
        if sub_path == "BUSI" or sub_path == "UDIAT":
            sub_path = "Breast-" + sub_path
        elif sub_path == "TN3K" or sub_path == "DDTI":
            sub_path = "ThyroidNodule-" + sub_path
        img_path = os.path.join(os.path.join(self.dataset_path, sub_path), 'img')
        label_path = os.path.join(os.path.join(self.dataset_path, sub_path), 'label')
        original_image = cv2.imread(os.path.join(img_path, filename + '.png'))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        mask = cv2.imread(os.path.join(label_path, filename + '.png'), cv2.IMREAD_UNCHANGED)
        # If mask is multi-channel, convert to single-channel grayscale
        if len(mask.shape) == 3 and mask.shape[2] > 1:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        elif len(mask.shape) == 2:
            pass  # Already grayscale
        else:
            raise ValueError(f"Unexpected mask format: {mask.shape}")

        classes = self.class_dict[sub_path]
        if classes == 2:
            mask[mask > 1] = 1
        original_image, mask = correct_dims(original_image, mask)
        if self.joint_transform:
            original_image, mask, low_mask = self.joint_transform(original_image, mask)
        mask = mask.unsqueeze(0)
        return {
            'image': original_image,
            'label': mask,
            'image_name': filename + '.png',
        }



class Logger:
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)
