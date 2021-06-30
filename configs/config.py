import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Folders' paths
TRAIN_IMAGES_PATH = '../gdrive/MyDrive/resized/'
TEST_IMAGES_PATH = './data/test/resized/'
TRAIN_CSV = './data/train.csv'
VALID_CSV = './data/validation.csv'
TEST_CSV = './data/test.csv'

# Hyper parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 32
NUM_EPOCHS = 5
NUM_WORKERS = 4
PERCENT_TRAIN = 0.8
PERCENT_VALID = 0.2
PRETRAINED = True
NUM_CLASSES = 1
REQUIRES_GRAD = True
OPTIMIZER = 'adam'
SAVE_MODEL = True
LOAD_MODEL = False
CHECKPOINT_FILE = './model/resnet50_epoch2.pth.tar'

# Data augmentation for images
transformations = {
    'train': A.Compose([
        A.Resize(width=230, height=230),
        A.RandomCrop(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Blur(p=0.3),
        A.ColorJitter(p=0.3),
        A.CoarseDropout(max_holes=12, max_height=20, max_width=20, p=0.3),
        # A.IAAAffine(shear=30, rotate=0, p=0.2, mode='constant'),
        A.Normalize(
            mean=[0.3200, 0.2241, 0.1608],
            std=[0.3025, 0.2186, 0.1742],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ]),
    'validation': A.Compose([
        A.Resize(height=230, width=230),
        A.Normalize(
            mean=[0.3200, 0.2241, 0.1608],
            std=[0.3025, 0.2186, 0.1742],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ]),
    'test': A.Compose([
        A.Resize(height=230, width=230),
        A.Normalize(
            mean=[0.3200, 0.2241, 0.1608],
            std=[0.3025, 0.2186, 0.1742],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
}

# Model classifier
model_fc = {
    'layer_1': 1024,
    'layer_2': NUM_CLASSES
}
