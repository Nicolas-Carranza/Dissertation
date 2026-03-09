# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
from torch.utils.data import Dataset


class MNISTDiscriDataset(Dataset):
    """
    Dataset for MNIST discrimination game.
    Each sample contains:
    - sender_input: the target image (flattened)
    - label: position of target (always 0 before shuffling)
    - receiver_input: all images [target + distractors] (flattened)
    """
    
    def __init__(self, mnist_dataset, n_distractors=2, seed=None):
        """
        Args:
            mnist_dataset: torchvision.datasets.MNIST instance
            n_distractors: number of distractor images per sample
            seed: random seed for reproducibility
        """
        self.mnist_data = mnist_dataset
        self.n_distractors = n_distractors
        self.n_items = n_distractors + 1  # target + distractors
        
        if seed is not None:
            random.seed(seed)
    
    def __len__(self):
        return len(self.mnist_data)
    
    def __getitem__(self, idx):
        # Get target image and label
        target_img, target_label = self.mnist_data[idx]
        
        # Get random distractor images (different from target)
        distractor_indices = random.sample(range(len(self.mnist_data)), self.n_distractors)
        distractor_imgs = [self.mnist_data[i][0] for i in distractor_indices]
        
        # Combine: [target] + [distractors]
        all_images = [target_img] + distractor_imgs
        
        # Stack and flatten
        # target_img: (1, 28, 28) -> (784,)
        sender_input = target_img.view(-1)
        
        # all_images: list of (1, 28, 28) -> (n_items, 784)
        receiver_input = torch.stack([img.view(-1) for img in all_images])
        
        # Shuffle positions so target isn't always first
        positions = list(range(self.n_items))
        random.shuffle(positions)
        
        # Find where target ended up after shuffle
        target_position = positions.index(0)
        receiver_input = receiver_input[positions]
        
        return sender_input, torch.tensor(target_position), receiver_input
