
from glob import glob
from dataset import DicomPairDataset, DicomToTensor
from model import TextureConversion

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms


if __name__ == "__main__":

    batch_size = 8
    epoch = 300
    save_path = "./models/BtoA"

    transform_list = [
        DicomToTensor()
    ]

    device = torch.device("cuda:0")
    #device = torch.device("cpu")

    model = TextureConversion()#.type(torch.DoubleTensor)
    model = model.to(device)

    dataset_FBP = DicomPairDataset(glob("E:/LiverCT/20210901-20211130_sort/testB/*/*.dcm"),
                                   glob("E:/LiverCT/20210901-20211130_sort/testA/*/*.dcm"),
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
