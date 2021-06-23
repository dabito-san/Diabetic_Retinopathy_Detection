# internal imports
from configs import config
from dataloaders.dataset import DRDataset
from model.resnet50 import Resnet50
from utils.util import get_accuracy, save_checkpoint

# external imports
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import cohen_kappa_score


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    """Trains the model for one epoch"""

    losses = []
    loop = tqdm(loader)
    for batch_idx, (x, y, _) in enumerate(loop):
        x = x.to(device=device)
        y = y.to(device=device)

        # forward pass
        with torch.cuda.amp.autocast():
            scores = model(x)
            loss = loss_fn(scores, y.unsqueeze(1).float())

        losses.append(loss.item())

        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    epoch_loss = sum(losses) / len(losses)

    return epoch_loss


def train_all_epochs():
    """Trains the model for all epochs"""

    # Create Datasets
    train_dataset = DRDataset(config.TRAIN_IMAGES_PATH,
                              config.TRAIN_CSV,
                              mode='train',
                              transform=config.transformations['train'])
    print('train_dataset created')
    valid_dataset = DRDataset(config.TRAIN_IMAGES_PATH,
                              config.VALID_CSV,
                              mode='validation',
                              transform=config.transformations['validation'])
    print('valid_dataset created')
    # TODO test_dataset

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
        shuffle=True
    )
    print('train_loader created')
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
        shuffle=False
    )
    print('valid_loader created')
    # TODO test_loader

    # # Create Loss function
    loss_fn = nn.MSELoss()
    print('loss_fn created')

    # Create model
    model = Resnet50()
    print('model created')
    model = model.to(device=config.DEVICE)
    print('model sent to device')
    optimizer = optim.Adam(model.parameters(),
                           lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    print('optimizer created')
    scaler = torch.cuda.amp.GradScaler()
    print('scaler created')

    # Train for all epochs
    for epoch in range(config.NUM_EPOCHS):
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)

        if config.SAVE_MODEL:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_checkpoint(checkpoint, filename=f'{model.model_name}_epoch{epoch}.pth.tar')

        # Get validation score
        preds, labels = get_accuracy(valid_loader, model, config.DEVICE)
        print(f'QuadraticWeightedKappa (Validation): {cohen_kappa_score(labels, preds, weights="quadratic")}')


if __name__ == '__main__':
    train_all_epochs()