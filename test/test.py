# internal imports
from configs import config
from model.resnet50 import Resnet50
from utils.util import load_checkpoint, make_prediction
from dataloaders.dataset import DRDataset

# external imports
import os
import torch
from torch import optim
from torch.utils.data import DataLoader


# Load checkpoint
if os.path.isfile(config.CHECKPOINT_FILE):
    model = Resnet50()
    print('model created')
    model = model.to(device=config.DEVICE)
    print(f'model sent to device: {config.DEVICE}')
    optimizer = optim.Adam(model.parameters(),
                           lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    print('optimizer created')

    load_checkpoint(torch.load(config.CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)
else:
    print('No checkpoint found')
    import sys
    sys.exit()

# Create Dataset
test_dataset = DRDataset(config.TEST_IMAGES_PATH,
                         config.TEST_CSV,
                         mode='test',
                         transform=config.transformations['test'])
print('test_dataset created')

# Create DataLoader
test_loader = DataLoader(
    test_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
    shuffle=False,
)
print('test_loader created')

# Make predictions
make_prediction(model, test_loader, output_csv='submission.csv')
