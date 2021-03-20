import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.checkpoint import checkpoint_sequential

import time
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-batch", "--batch_size", help="Size of batch used during training",
                    type=int, default=128)
parser.add_argument("-seed", "--random_seed", help="random seed used in data preparation",
                    type=int, default=42)
parser.add_argument("-epochs", "--num_epochs", help="Number of training epochs",
                    type=int, default=100)
parser.add_argument("--num_checkpoint_parts", help="Number of training epochs",
                    type=int)

parser.add_argument("train_path", help="path to train dataset", type=str)
parser.add_argument("test_path", help="path to test dataset", type=str)
parser.add_argument("--save_model_path", help="path where model should be saved", type=str)

args = parser.parse_args()


def get_data_loaders(train_data_path, test_data_path, batch_size=128, random_seed=42):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = torchvision.datasets.ImageFolder(train_data_path, transform=transform_train)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [80000, 20000],
                                                               generator=torch.Generator().manual_seed(random_seed))
    test_dataset = torchvision.datasets.ImageFolder(test_data_path, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


class Vgg11LikeModel(nn.Module):

    def __init__(self, n_classes=200):
        super(Vgg11LikeModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [batch_size, 512, 4, 4]

            # [batch_size, 8192]
            nn.Flatten(),

            nn.Linear(8192, 4096),
            nn.ReLU(),

            nn.Linear(4096, 4096),
            nn.ReLU(),

            nn.Linear(4096, n_classes)
        )

    def forward(self, X):
        return self.layers(X)


def train_model(model, loaders, n_epochs, checkpoint_path):
    train_loader, val_loader, test_loader = loaders
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"device: {device}")

    model.train()
    model.to(device)

    lr = 0.002
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epoch_time = []
    batch_time = []

    for epoch in range(n_epochs):
        epoch_start_time = time.perf_counter()
        if epoch and not epoch % 10 and checkpoint_path is not None:
            torch.save(model.state_dict(), checkpoint_path)
        if epoch and not epoch % 50:
            lr /= 10
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for i, data in enumerate(train_loader):

            inputs, labels = data[0].to(device), data[1].to(device)

            batch_start_time = time.perf_counter()
            optimizer.zero_grad()
            inputs.requires_grad = True
            if args.num_checkpoint_parts is None:
                outputs = model(inputs)
            else:
                outputs = checkpoint_sequential(model.layers, args.num_checkpoint_parts, inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_time.append(time.perf_counter() - batch_start_time)

        epoch_time.append(time.perf_counter() - epoch_start_time)

    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)
    return epoch_time, batch_time


if __name__ == "__main__":

    loaders = get_data_loaders(
        args.train_path, args.test_path,
        batch_size=args.batch_size,
        random_seed=args.random_seed
    )
    model = Vgg11LikeModel()
    if args.num_checkpoint_parts is not None and args.num_checkpoint_parts > len(model.layers):
        raise RuntimeError(f"Number of checkpoint segments is bigger than number of layers of model"
                           f"({args.num_checkpoint_parts} > {len(model.layers)})")

    start_training_time = time.perf_counter()
    epoch_time, batch_time = train_model(model, loaders, args.num_epochs, args.save_model_path)
    overall_time = time.perf_counter() - start_training_time
    memory_usage_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.eval()
    model.to(device)

    correct_answers_val, num_samples_val = 0.0, 0.0
    with torch.no_grad():
        for data_val in loaders[1]:
            inputs, labels = data_val[0].to(device), data_val[1].cpu()
            outputs = model(inputs).cpu()
            correct_answers_val += sum(outputs.argmax(dim=1) == labels)
            num_samples_val += len(inputs)
    val_acc = correct_answers_val / num_samples_val

    correct_answers_test, num_samples_test = 0.0, 0.0
    with torch.no_grad():
        for data_test in loaders[2]:
            inputs, labels = data_test[0].to(device), data_test[1].cpu()
            outputs = model(inputs).cpu()
            correct_answers_test += sum(outputs.argmax(dim=1) == labels)
            num_samples_test += len(inputs)
    test_acc = correct_answers_test / num_samples_test

    result_str = f"Overall time = {round(overall_time / 60, 8)} minutes, " \
        f"num epochs = {len(epoch_time)}\n" \
        f"Number checkpoint segments = {args.num_checkpoint_parts}\n"\
        f"mean epoch time = {round(float(np.mean(epoch_time)), 8)} seconds, "\
        f"mean_batch time = {round(float(np.mean(batch_time)), 8)} seconds\n"\
        f"Peak memory usage by Pytorch tensors: {memory_usage_mb:.2f} Mb\n"\
        f"Validation set accuracy: {val_acc}\n" \
        f"Test set accuracy: {test_acc}\n" \
        "training is finished\n________________\n"

    print(result_str)
    with open(f"results_{args.num_checkpoint_parts}_segments_{args.num_epochs}_epochs.txt", "w") as f:
        f.write(result_str)
