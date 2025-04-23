#!/usr/bin/env python3

import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from util.io import load_json

from .transform import (
    RandomGaussianNoise,
    RandomHorizontalFlipFLow,
    RandomOffsetFlow,
    SeedableRandomSquareCrop,
    ThreeCrop,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FrameReader:

    IMG_NAME = '{:06d}.jpg'

    def __init__(self, frame_dir, modality, crop_transform, img_transform,
                 same_transform, second_stream_transforms=None):
        self._frame_dir = frame_dir
        self._pose_dir = os.path.join(os.path.dirname(frame_dir), 'mediapipe_hand_pose')
        # self._pose_dir = os.path.join(os.path.dirname(frame_dir), 'precompute_pose_heatmap')
        self._modality = modality
        self._is_flow = modality == 'flow'
        self._crop_transform = crop_transform
        self._img_transform = img_transform
        self._same_transform = same_transform
        self._second_stream_transforms = second_stream_transforms

        # Preload the pose json file
        self._pose_data = {}
        for js_file in os.listdir(self._pose_dir):
            if js_file.endswith('.json'):
                with open(os.path.join(self._pose_dir, js_file), 'r') as f:
                    self._pose_data[js_file] = json.load(f)

        # Preload the Gaussian kernel
        sigma = 3
        sigma_sq_2 = 2 * sigma * sigma
        kernel_size = int(6 * sigma + 3)
        radius = kernel_size // 2

        # Precompute Gaussian kernel
        y = torch.arange(-radius, radius + 1, device='cpu', dtype=torch.float32)
        x = torch.arange(-radius, radius + 1, device='cpu', dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        gaussian = torch.exp(-(xx**2 + yy**2) / sigma_sq_2)

        self._radius = radius
        self._gaussian = gaussian

    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path).float() / 255
        if self._is_flow:
            img = img[1:, :, :]     # GB channels contain data
        return img

    def read_rgb_flow(self, frame_path):
        # Assume RGB is in frames/ and Flow is in flows/
        img = torchvision.io.read_image(frame_path).float() / 255
        flow = torchvision.io.read_image(frame_path.replace('frames', 'flows')).float() / 255
        flow = flow[1:, :, :]     # GB channels contain data

        return img, flow

    def read_rgb_pose(self, frame_path):
        raise NotImplementedError('Pose not implemented yet!')

    def gaussian_pose(self, pose, size, dtype=torch.float32, sigma=3):
        heatmap = torch.zeros((42, *size), dtype=dtype, device='cpu')
        height, width = size
        # sigma_sq_2 = 2 * sigma * sigma
        # kernel_size = int(6 * sigma + 3)
        # radius = kernel_size // 2

        # # Precompute Gaussian kernel
        # y = torch.arange(-radius, radius + 1, device='cpu', dtype=dtype)
        # x = torch.arange(-radius, radius + 1, device='cpu', dtype=dtype)
        # yy, xx = torch.meshgrid(y, x, indexing='ij')
        # gaussian = torch.exp(-(xx**2 + yy**2) / sigma_sq_2)

        def draw_keypoints(keypoints, offset):
            for idx, (x_center, y_center) in enumerate(keypoints):
                if x_center is None or y_center is None:
                    continue
                x_center, y_center = int(x_center), int(y_center)
                if not (0 <= x_center < width and 0 <= y_center < height):
                    continue

                x0 = max(0, x_center - self._radius)
                y0 = max(0, y_center - self._radius)
                x1 = min(width, x_center + self._radius + 1)
                y1 = min(height, y_center + self._radius + 1)

                g_x0 = self._radius - (x_center - x0)
                g_y0 = self._radius - (y_center - y0)
                g_x1 = g_x0 + (x1 - x0)
                g_y1 = g_y0 + (y1 - y0)

                heatmap[offset + idx, y0:y1, x0:x1] = torch.maximum(
                    heatmap[offset + idx, y0:y1, x0:x1],
                    self._gaussian[g_y0:g_y1, g_x0:g_x1]
                )

        if 'left' in pose:
            draw_keypoints(pose['left'], offset=0)
        else:
            heatmap[:21, :, :] = 0.05

        if 'right' in pose:
            draw_keypoints(pose['right'], offset=21)
        else:
            heatmap[21:, :, :] = 0.05

        return heatmap

    def load_frames(self, video_name, start, end, pad=False, stride=1,
                    randomize=False):
        rand_crop_state = None
        rand_state_backup = None
        ret = []
        n_pad_start = 0
        n_pad_end = 0
        # with open(os.path.join(self._pose_dir, f'{video_name}.json'), 'r') as f:
        #     pose_data = json.load(f)

        pose_data = self._pose_data.get(video_name, {})

        # Load directly from torch precomputed tensor
        # pose_data = torch.load(os.path.join(self._pose_dir, f'{video_name}.pt')).to_dense()

        # start_time = time.time()
        for frame_num in range(start, end, stride):
            if randomize and stride > 1:
                frame_num += random.randint(0, stride - 1)

            if frame_num < 0:
                n_pad_start += 1
                continue

            frame_path = os.path.join(
                self._frame_dir, video_name,
                FrameReader.IMG_NAME.format(frame_num))
            try:
                if self._modality == 'twostream':
                    img, second = self.read_rgb_flow(frame_path)
                else:
                    img = self.read_frame(frame_path) # (3, H, W)
                    if os.path.basename(frame_path) in pose_data:
                        _data = pose_data[os.path.basename(frame_path)]
                    else:
                        _data = {}

                    start_time = time.time()
                    second = self.gaussian_pose(_data, list(img.shape[-2:])) # (42, H, W)
                    print("DEBUG >>> Load pose", time.time() - start_time)
                    # second = pose_data[frame_num] # (42, H, W)

                if self._crop_transform:
                    if self._same_transform:
                        if rand_crop_state is None:
                            rand_crop_state = random.getstate()
                        else:
                            rand_state_backup = random.getstate()
                            random.setstate(rand_crop_state)

                    # Need to combine before cropping
                    img = self._crop_transform(img)

                    # Restore the same seed and crop
                    if rand_crop_state is not None:
                        # During evaluation/inference, CenterCrop is used instead of RandomCrop
                        random.setstate(rand_crop_state)
                    second = self._crop_transform(second) # In pose, second always exists

                    # if second is not None:
                    #     img = torch.cat((img, second), dim=0)
                    #     img = self._crop_transform(img)
                    #     second = img[3:, :, :].clone()  # Copy pose heatmap
                    #     img = img[:3, :, :].clone()  # Only keep RGB channels
                    # else:
                    #     img = self._crop_transform(img)
                    # # second = self._crop_transform(second) if second is not None else None

                    if rand_state_backup is not None:
                        # Make sure that rand state still advances
                        random.setstate(rand_state_backup)
                        rand_state_backup = None

                if not self._same_transform:
                    img = self._img_transform(img)
                    # second_stream_transforms can be empty list -> Not using it
                    if self._second_stream_transforms:
                        second = self._second_stream_transforms(second) if second is not None else None

                if second is not None:
                    # Merge RGB and Pose together
                    img = torch.cat((img, second), dim=0)

                ret.append(img)

                ###### Visualize debug #####
                # import matplotlib.pyplot as plt

                # # Save RGB image (first 3 channels)
                # rgb_img = (img[:3, :, :].clamp(0, 1) * 255).byte()
                # torchvision.io.write_jpeg(rgb_img, f"{video_name}_{frame_num}_img_debug.jpg")

                # # Create and normalize pose heatmap from remaining channels
                # debug_pose = img[3:, :, :].sum(dim=0)
                # debug_pose = (debug_pose - debug_pose.min()) / (debug_pose.max() - debug_pose.min() + 1e-5)  # avoid divide by 0

                # # Save heatmap as clean image
                # plt.imshow(debug_pose.cpu().numpy(), cmap='inferno')
                # plt.axis('off')
                # plt.savefig(f"{video_name}_{frame_num}_pose_debug.jpg", bbox_inches='tight', pad_inches=0)
                # plt.close()
                ###### End #####
            except RuntimeError as e:
                # print("DEBUG", e)
                # print('Missing file!', frame_path)
                n_pad_end += 1

        # In the multicrop case, the shape is (B, T, C, H, W)
        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4)) # DEBUG (40, 5, 224, 224)
        if self._same_transform:
            if self._modality == 'twostream':
                ret[:, :3, :, :] = self._img_transform(ret[:, :3, :, :]) # the first 3 channels are RGB
                ret[:, 3:, :, :] = self._second_stream_transforms(ret[:, 3:, :, :]) # the last 2 channels are Flow
            else:
                # ret = self._img_transform(ret) # TODO split transform here.
                # No Flip transform here.
                ret[:, :3, :, :] = self._img_transform(ret[:, :3, :, :]) # the first 3 channels are RGB
                ret[:, 3:, :, :] = self._second_stream_transforms(ret[:, 3:, :, :]) # the last 42 channels are Pose

        # Always pad start, but only pad end if requested
        if n_pad_start > 0 or (pad and n_pad_end > 0):
            ret = nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad else 0))

        # print("DEBUG >>> Load frames", time.time() - start_time)
        return ret
        # return ret.type(torch.float16) # try to save memory


# Pad the start/end of videos with empty frames
DEFAULT_PAD_LEN = 5

def _get_deferred_rgb_transform():
    img_transforms = [
        # Jittering separately is faster (low variance)
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([
                transforms.ColorJitter(saturation=(0.7, 1.2))
            ]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([
                transforms.ColorJitter(brightness=(0.7, 1.2))
            ]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([
                transforms.ColorJitter(contrast=(0.7, 1.2))
            ]), p=0.25),

        # Jittering together is slower (high variance)
        # transforms.RandomApply(
        #     nn.ModuleList([
        #         transforms.ColorJitter(
        #             brightness=(0.7, 1.2), contrast=(0.7, 1.2),
        #             saturation=(0.7, 1.2), hue=0.2)
        #     ]), 0.8),

        transforms.RandomApply(
            nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
    return torch.jit.script(nn.Sequential(*img_transforms))


def _get_deferred_bw_transform():
    img_transforms = [
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(brightness=0.3)]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(contrast=0.3)]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        RandomGaussianNoise()
    ]
    return torch.jit.script(nn.Sequential(*img_transforms))


def _load_frame_deferred(gpu_transform, batch, device):
    frame = batch['frame'].to(device) # (B, T, C, H, W)
    with torch.no_grad():
        for i in range(frame.shape[0]):
            # frame[i] = gpu_transform(frame[i])
            frame[i][:, :3, :, :] = gpu_transform(frame[i][:, :3, :, :]) # Only RGB channels

        if 'mix_weight' in batch:
            weight = batch['mix_weight'].to(device)
            frame *= weight[:, None, None, None, None]

            frame_mix = batch['mix_frame']
            for i in range(frame.shape[0]):
                frame[i] += (1. - weight[i]) * gpu_transform(
                    frame_mix[i].to(device))
    return frame

def _rgb_transforms(is_eval, defer_transform):
    img_transforms = []
    if not is_eval:
        # img_transforms.append(
        #     transforms.RandomHorizontalFlip()) # Not make sense in pose, left and right hand is swapped

        if not defer_transform:
            img_transforms.extend([
                # Jittering separately is faster (low variance)
                transforms.RandomApply(
                    nn.ModuleList([transforms.ColorJitter(hue=0.2)]),
                    p=0.25),
                transforms.RandomApply(
                    nn.ModuleList([
                        transforms.ColorJitter(saturation=(0.7, 1.2))
                    ]), p=0.25),
                transforms.RandomApply(
                    nn.ModuleList([
                        transforms.ColorJitter(brightness=(0.7, 1.2))
                    ]), p=0.25),
                transforms.RandomApply(
                    nn.ModuleList([
                        transforms.ColorJitter(contrast=(0.7, 1.2))
                    ]), p=0.25),

                transforms.RandomApply(
                    nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25)
            ])

    if not defer_transform:
        img_transforms.append(transforms.Normalize(
            mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return img_transforms

def _get_img_transforms(
        is_eval,
        crop_dim,
        modality,
        same_transform,
        defer_transform=False,
        multi_crop=False
):
    crop_transform = None
    if crop_dim is not None:
        if multi_crop:
            assert is_eval
            crop_transform = ThreeCrop(crop_dim)
        elif is_eval:
            crop_transform = transforms.CenterCrop(crop_dim)
        elif same_transform:
            print('=> Using seeded crops!')
            crop_transform = SeedableRandomSquareCrop(crop_dim)
        else:
            crop_transform = transforms.RandomCrop(crop_dim)

    img_transforms = []
    second_stream_transforms = [] # use in RGB + Flow or RGB + Pose

    if modality == 'rgb':
        img_transforms = _rgb_transforms(is_eval, defer_transform)

    elif modality == 'bw':
        if not is_eval:
            img_transforms.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25)])
        img_transforms.append(transforms.Grayscale())

        if not defer_transform:
            if not is_eval:
                img_transforms.extend([
                    transforms.RandomApply(
                        nn.ModuleList([transforms.ColorJitter(brightness=0.3)]),
                        p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([transforms.ColorJitter(contrast=0.3)]),
                        p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
                ])

            img_transforms.append(transforms.Normalize(
                mean=[0.5], std=[0.5]))

            if not is_eval:
                img_transforms.append(RandomGaussianNoise())
    elif modality == 'flow':
        assert not defer_transform

        img_transforms.append(transforms.Normalize(
            mean=[0.5, 0.5], std=[0.5, 0.5]))

        if not is_eval:
            img_transforms.extend([
                RandomHorizontalFlipFLow(),
                RandomOffsetFlow(),
                RandomGaussianNoise()
            ])
    elif modality == 'twostream':
        # Basically combined RGB and Flow transforms above
        assert not defer_transform

        # RGB transforms
        img_transforms = _rgb_transforms(is_eval, defer_transform)

        # Flow transforms
        second_stream_transforms.append(transforms.Normalize(
            mean=[0.5, 0.5], std=[0.5, 0.5]))

        if not is_eval:
            second_stream_transforms.extend([
                RandomHorizontalFlipFLow(),
                RandomOffsetFlow(),
                RandomGaussianNoise()
            ])
    elif modality == 'pose':
        img_transforms = _rgb_transforms(is_eval, defer_transform) # comment out random flip now
        second_stream_transforms.append(transforms.Normalize(mean=[0.5]*42, std=[0.5]*42))
        # Since all the transforms on RGB are about color, and crop is handled separately -> No need to trasnform pose heatmap.
        # raise NotImplementedError(modality)

    else:
        raise NotImplementedError(modality)

    img_transform = torch.jit.script(nn.Sequential(*img_transforms))
    if second_stream_transforms:
        second_stream_transforms = torch.jit.script(
            nn.Sequential(*second_stream_transforms))

    return crop_transform, img_transform, second_stream_transforms


def _print_info_helper(src_file, labels):
        num_frames = sum([x['num_frames'] for x in labels])
        num_events = sum([len(x['events']) for x in labels])
        print('{} : {} videos, {} frames, {:0.5f}% non-bg'.format(
            src_file, len(labels), num_frames,
            num_events / num_frames * 100))


IGNORED_NOT_SHOWN_FLAG = False


class ActionSpotDataset(Dataset):

    def __init__(
        self,
        classes,  # dict of class names to idx
        label_file,  # path to label json
        frame_dir,  # path to frames
        modality,  # [rgb, bw, flow] # add twostream
        clip_len,
        dataset_len,  # Number of clips
        is_eval=True,  # Disable random augmentation
        crop_dim=None,
        stride=1,  # Downsample frame rate
        same_transform=True,  # Apply the same random augmentation to
        # each frame in a clip
        dilate_len=0,  # Dilate ground truth labels
        mixup=False,
        pad_len=DEFAULT_PAD_LEN,  # Number of frames to pad the start
        # and end of videos
        fg_upsample=-1,  # Sample foreground explicitly
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}

        # Sample videos weighted by their length
        num_frames = [v['num_frames'] for v in self._labels]
        self._weights_by_length = np.array(num_frames) / np.sum(num_frames)

        self._clip_len = clip_len
        assert clip_len > 0
        self._stride = stride
        assert stride > 0
        self._dataset_len = dataset_len
        assert dataset_len > 0
        self._pad_len = pad_len
        assert pad_len >= 0
        self._is_eval = is_eval

        # Label modifications
        self._dilate_len = dilate_len
        self._fg_upsample = fg_upsample

        # Sample based on foreground labels
        if self._fg_upsample > 0:
            self._flat_labels = []
            for i, x in enumerate(self._labels):
                for event in x['events']:
                    if event['frame'] < x['num_frames']:
                        self._flat_labels.append((i, event['frame']))

        self._mixup = mixup

        # Try to do defer the latter half of the transforms to the GPU
        self._gpu_transform = None
        if not is_eval and same_transform:
            if modality == 'rgb':
                print('=> Deferring some RGB transforms to the GPU!')
                self._gpu_transform = _get_deferred_rgb_transform()
            elif modality == 'bw':
                print('=> Deferring some BW transforms to the GPU!')
                self._gpu_transform = _get_deferred_bw_transform()

        crop_transform, img_transform, second_stream_transforms = _get_img_transforms(
            is_eval, crop_dim, modality, same_transform,
            defer_transform=self._gpu_transform is not None)

        self._frame_reader = FrameReader(
            frame_dir, modality, crop_transform, img_transform, same_transform, second_stream_transforms)

    def load_frame_gpu(self, batch, device):
        if self._gpu_transform is None:
            frame = batch['frame'].to(device)
        else:
            frame = _load_frame_deferred(self._gpu_transform, batch, device)
        return frame

    def _sample_uniform(self):
        video_meta = random.choices(
            self._labels, weights=self._weights_by_length)[0]

        video_len = video_meta['num_frames']
        base_idx = -self._pad_len * self._stride + random.randint(
            0, max(0, video_len - 1
                       + (2 * self._pad_len - self._clip_len) * self._stride))
        return video_meta, base_idx

    def _sample_foreground(self):
        video_idx, frame_idx = random.choices(self._flat_labels)[0]
        video_meta = self._labels[video_idx]
        video_len = video_meta['num_frames']

        lower_bound = max(
            -self._pad_len * self._stride,
            frame_idx - self._clip_len * self._stride + 1)
        upper_bound = min(
            video_len - 1 + (self._pad_len - self._clip_len) * self._stride,
            frame_idx)

        base_idx = random.randint(lower_bound, upper_bound) \
            if upper_bound > lower_bound else lower_bound

        assert base_idx <= frame_idx
        assert base_idx + self._clip_len > frame_idx
        return video_meta, base_idx

    def _get_one(self):
        if self._fg_upsample > 0 and random.random() >= self._fg_upsample:
            video_meta, base_idx = self._sample_foreground()
        else:
            video_meta, base_idx = self._sample_uniform()

        labels = np.zeros(self._clip_len, np.int64)
        for event in video_meta['events']:
            event_frame = event['frame']

            # Index of event in label array
            label_idx = (event_frame - base_idx) // self._stride
            if (label_idx >= -self._dilate_len
                and label_idx < self._clip_len + self._dilate_len
            ):
                label = self._class_dict[event['label']]
                for i in range(
                    max(0, label_idx - self._dilate_len),
                    min(self._clip_len, label_idx + self._dilate_len + 1)
                ):
                    labels[i] = label

        frames = self._frame_reader.load_frames(
                        video_meta['video'],
                        base_idx,
                        base_idx + self._clip_len * self._stride,
                        pad=True,
                        stride=self._stride,
                        randomize=not self._is_eval # TODO not sure the purpose of this one
                        )

        return {'frame': frames, 'contains_event': int(np.sum(labels) > 0),
                'label': labels}

    def __getitem__(self, unused):
        ret = self._get_one()
        # print("DEBUG >>> Get one clip", ret['frame'].shape)
        if self._mixup:
            mix = self._get_one()    # Sample another clip
            l = random.betavariate(0.2, 0.2)
            label_dist = np.zeros((self._clip_len, len(self._class_dict) + 1))
            label_dist[range(self._clip_len), ret['label']] = l
            label_dist[range(self._clip_len), mix['label']] += 1. - l

            if self._gpu_transform is None:
                ret['frame'] = l * ret['frame'] + (1. - l) * mix['frame']
            else:
                ret['mix_frame'] = mix['frame']
                ret['mix_weight'] = l

            ret['contains_event'] = max(
                ret['contains_event'], mix['contains_event'])
            ret['label'] = label_dist

        return ret

    def __len__(self):
        return self._dataset_len

    def print_info(self):
        _print_info_helper(self._src_file, self._labels)


class ActionSpotVideoDataset(Dataset):

    def __init__(
            self,
            classes,
            label_file,
            frame_dir,
            modality,
            clip_len,
            overlap_len=0,
            crop_dim=None,
            stride=1,
            pad_len=DEFAULT_PAD_LEN,
            flip=False,
            multi_crop=False,
            skip_partial_end=True
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._clip_len = clip_len
        self._stride = stride

        crop_transform, img_transform, second_stream_transforms = _get_img_transforms(
            is_eval=True, crop_dim=crop_dim, modality=modality, same_transform=True, multi_crop=multi_crop)

        # No need to enforce same_transform since the transforms are
        # deterministic
        self._frame_reader = FrameReader(
            frame_dir, modality, crop_transform, img_transform, False, second_stream_transforms)

        self._flip = flip
        self._multi_crop = multi_crop

        self._clips = []
        for l in self._labels:
            has_clip = False
            for i in range(
                -pad_len * self._stride,
                max(0, l['num_frames'] - (overlap_len * stride)
                        * int(skip_partial_end)), \
                # Need to ensure that all clips have at least one frame
                (clip_len - overlap_len) * self._stride
            ):
                has_clip = True
                self._clips.append((l['video'], i))
            assert has_clip, l

    def __len__(self):
        return len(self._clips)

    def __getitem__(self, idx):
        video_name, start = self._clips[idx]
        frames = self._frame_reader.load_frames(
            video_name, start, start + self._clip_len * self._stride, pad=True,
            stride=self._stride)

        if self._flip:
            frames = torch.stack((frames, frames.flip(-1)), dim=0)

        return {'video': video_name, 'start': start // self._stride,
                'frame': frames}

    def get_labels(self, video):
        meta = self._labels[self._video_idxs[video]]
        num_frames = meta['num_frames']
        num_labels = num_frames // self._stride
        if num_frames % self._stride != 0:
            num_labels += 1
        labels = np.zeros(num_labels, np.int)
        for event in meta['events']:
            frame = event['frame']
            if frame < num_frames:
                labels[frame // self._stride] = self._class_dict[event['label']]
            else:
                print('Warning: {} >= {} is past the end {}'.format(
                    frame, num_frames, meta['video']))
        return labels

    @property
    def augment(self):
        return self._flip or self._multi_crop

    @property
    def videos(self):
        return sorted([
            (v['video'], v['num_frames'] // self._stride,
             v['fps'] / self._stride) for v in self._labels])

    @property
    def labels(self):
        assert self._stride > 0
        if self._stride == 1:
            return self._labels
        else:
            labels = []
            for x in self._labels:
                x_copy = copy.deepcopy(x)
                x_copy['fps'] /= self._stride
                x_copy['num_frames'] //= self._stride
                for e in x_copy['events']:
                    e['frame'] //= self._stride
                labels.append(x_copy)
            return labels

    def print_info(self):
        num_frames = sum([x['num_frames'] for x in self._labels])
        num_events = sum([len(x['events']) for x in self._labels])
        print('{} : {} videos, {} frames ({} stride), {:0.5f}% non-bg'.format(
            self._src_file, len(self._labels), num_frames, self._stride,
            num_events / num_frames * 100))
