from glob import glob
from dataset import DicomPairDataset, DicomToTensor
from model import TextureConversion
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms


def train(opt):
    batch_size = opt.batch
    epoch = opt.epoch
    save_path = opt.save_path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    transform_list = [
        DicomToTensor()
    ]

    device = torch.device("cpu" if opt.device == -1 else opt.device)
    # device = torch.device("cpu")

    model = TextureConversion()  # .type(torch.DoubleTensor)
    model = model.to(device)

    dataset_FBP = DicomPairDataset(glob(opt.dataA_path + "/**/*.dcm", recursive=True),
                                   glob(opt.dataB_path + "/**/*.dcm", recursive=True),
                                   transforms.Compose(transform_list))
    train_loader = DataLoader(dataset_FBP, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _epoch in range(epoch):
        running_loss = 0.0

        for i, target in enumerate(train_loader):
            # get the inputs
            data, label = target["data"].to(device), target["label"].to(device)
            input_data = data.clone()

            # compute output
            output = model(data)
            loss = criterion(output + input_data, label)

            # compute gradient and do Adam step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %f' %
                      (_epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

        if _epoch % 10 == 9:
            torch.save(model.state_dict(), save_path + "/{0:03d}_TC.pth".format(_epoch + 1))
        torch.save(model.state_dict(), save_path + "/last_TC.pth")

    print('Finished Training')


def train_arguments():
    opt = argparse.ArgumentParser()

    opt.add_argument("--dataA_path", type=str, help="dataset-A directory path. It extracts sub-directories.")
    opt.add_argument("--dataB_path", type=str, help="dataset-B directory path. It extracts sub-directories.")
    opt.add_argument("--save_path", type=str, help="trained model save path")

    opt.add_argument("--device", type=int, help="gpu ids: e.g. 0 or 1. use -1 for CPU")
    opt.add_argument("--batch", type=int, help="batch size")
    opt.add_argument("--epoch", type=int, help="epoch size")

    return opt


if __name__ == "__main__":
    parser = train_arguments()
    train(parser.parse_args())
