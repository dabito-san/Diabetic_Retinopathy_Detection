import numpy as np
from numpy.random import RandomState
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from configs import config
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import os


def get_images_mean_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _, _ in dataloader:
        channels_sum += torch.mean(data.float(), dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data.float() ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def train_valid_split(csv_path, save_path):
    data = pd.read_csv(csv_path)
    rng = RandomState()

    train = data.sample(frac=config.PERCENT_TRAIN, random_state=rng)
    valid = data.loc[~data.index.isin(train.index)]

    train.to_csv(save_path + 'train.csv', index=False)
    valid.to_csv(save_path + 'validation.csv', index=False)

    print(f'Total number of images: {data.shape[0]}')
    print(f'Number of training images: {train.shape[0]}')
    print(f'Number of validation images: {valid.shape[0]}')


def test_set(test_images_path, csv_save_path):
    images = []
    for image in os.listdir(test_images_path):
        images.append(str(image))
    images = pd.DataFrame(images, columns=['image'])
    images.to_csv(csv_save_path + 'test.csv', index=False)


def get_accuracy(loader, model, device='cuda'):
    model.eval()
    num_correct = 0
    num_samples = 0
    all_preds, all_labels = [], []

    for x, y, _ in loader:
        x = x.to(device=device)
        y = y.to(device=device)

        predictions = model.forward(x)
        predictions[predictions < 0.5] = 0
        predictions[(predictions >= 0.5) & (predictions < 1.5)] = 1
        predictions[(predictions >= 1.5) & (predictions < 2.5)] = 2
        predictions[(predictions >= 2.5) & (predictions < 3.5)] = 3
        predictions[(predictions >= 3.5)] = 4
        predictions = predictions.long().view(-1)
        y = y.view(-1)

        num_correct += (predictions == y).sum()
        num_samples += predictions.shape[0]

        all_preds.append(predictions.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    print(f'[{num_correct}/{num_samples}] sample predictions are correct')
    print(f'Accuracy: {float(num_correct) / float(num_samples) * 100:.2f}%')

    model.train()

    return np.concatenate(all_preds, axis=0), np.concatenate(
        all_labels, axis=0)


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('Saving checkpoint...')
    torch.save(state, filename)
    print('=> Checkpoint saved.')


def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    #optimizer.load_state_dict(checkpoint["optimizer"])

    # Reset learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def create_sampler(labels_list):
    labels_count = {
        '0': 0,
        '1': 0,
        '2': 0,
        '3': 0,
        '4': 0
    }
    for label in labels_list:
        labels_count[str(label)] += 1

    classes_weights = [v / len(labels_list) for v in labels_count.values()]
    samples_weights = torch.from_numpy(np.array([classes_weights[l] for l in labels_list]))

    # Instanciate sampler object
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    return sampler


def make_prediction(model, loader, output_csv="submission.csv"):
    preds = []
    filenames = []
    model.eval()

    for x, y, files in loader:
        x = x.to(config.DEVICE)
        with torch.no_grad():
            predictions = model(x)

            predictions[predictions < 0.5] = 0
            predictions[(predictions >= 0.5) & (predictions < 1.5)] = 1
            predictions[(predictions >= 1.5) & (predictions < 2.5)] = 2
            predictions[(predictions >= 2.5) & (predictions < 3.5)] = 3
            predictions[(predictions >= 3.5)] = 4

            predictions = predictions.long().view(-1)
            preds.append(predictions.cpu().numpy())
            filenames += files

    df = pd.DataFrame({"image": filenames, "level": np.concatenate(preds, axis=0)})
    df.to_csv(output_csv, index=False)
    model.train()
    print("Done with predictions")


if __name__ == '__main__':
    from dataloaders.dataset import *

    dataset = DRDataset(images_folder='../data/train/',
                        csv_path='../data/trainLabels.csv',
                        mode='validation',
                        transform=config.transformations['validation'])
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)

    mean, std = get_images_mean_std(dataloader)
    print(mean)
    print(std)
