from configs import config
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm


class DRDataset(Dataset):
    def __init__(self, images_folder, csv_path, mode, transform=None):
        super().__init__()
        self.data = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.images_files = os.listdir(images_folder)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'test':
            image_file, label = self.data.iloc[index], -1
        else:
            image_file, label = self.data.iloc[index]

        image = np.array(Image.open(os.path.join(self.images_folder, image_file+'.jpeg')))

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, label, image_file


if __name__ == '__main__':
    """
    Test if everything works fine
    """
    train_dataset = DRDataset(
        images_folder='../data/train/',
        csv_path='../data/trainLabels.csv',
        mode='validation',
        transform=config.transformations['validation']
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )

    for image, label, file in tqdm(train_loader):
        print(image.shape)
        print(label.shape)

        import sys

        sys.exit()