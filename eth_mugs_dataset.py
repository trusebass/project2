"""ETH Mugs Dataset."""

import os
from PIL import Image
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

from utils import IMAGE_SIZE, load_mask


# This is only an example - you DO NOT have to use it exactly in this form!
class ETHMugsDataset(Dataset):
    """Torch dataset template shared as an example."""

    #Note: mode = train should be changed to just mode, otherwise will be train even when test arg passed
    def __init__(self, root_dir, mode):
        """This dataset class loads the ETH Mugs dataset.

        It will return the resized image according to the scale and mask tensors
        in the original resolution.

        Args:
            root_dir (str): Path to the root directory of the dataset.
            mode (str): Mode of the dataset. It can be "train", "val" or "test"
        """
        self.mode = mode
        self.root_dir = root_dir

        # TODO: get image and mask paths
        self.rgb_dir = os.path.join(self.root_dir, "rgb").replace("/","\\")
        self.mask_dir = os.path.join(self.root_dir, "masks").replace("/","\\")
        self.image_paths = os.listdir(self.rgb_dir)
        self.mask_paths = os.listdir(self.mask_dir)
        # TODO: set image transforms - these transforms will be applied to pre-process the data before passing it through the model
        self.transform = None  # TO-DO

        print("[INFO] Dataset mode:", mode)
        print(
            "[INFO] Number of images in the {} dataset: {}".format(mode, len(self.image_paths))
        )

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Get an item from the dataset."""
        # TODO: load image and gt mask (unless when in test mode), apply transforms if necessary
        '''
        if self.mode == "test":
            mask_path = os.path.join (self.mask_dir, self.image_paths[idx])
            mask = read_image(mask_path, ImageReadMode = ImageReadMode.UNCHANGED)
            return mask
        '''     
        mask_path = os.path.join (self.mask_dir, self.mask_paths[idx])
        mask = read_image(mask_path)
        
        img_path = os.path.join(self.rgb_dir, self.image_paths[idx])
        image = read_image(img_path)
        return image, mask
