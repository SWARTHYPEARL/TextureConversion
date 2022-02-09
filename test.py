
import os
from glob import glob
from dataset import DicomTestDataset, DicomToTensor
from model import TextureConversion
from util import tensor2dicom

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def test(load_model_path: str, dataset_A: str, dataset_B: str, save_path: str):

    #load_model_path = "./models/last_TC - 복사본.pth"
    #dataset_A = "E:/LiverCT/20210901-20211130_sort/testA/10952379_20210925"
    #dataset_B = "E:/LiverCT/20210901-20211130_sort/testB/10952379_20210925"
    #save_path = "./results"

    transform_list = [
        DicomToTensor()
    ]

    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    model = TextureConversion()  # .type(torch.DoubleTensor)
    model.load_state_dict(torch.load(load_model_path, map_location=device))
    model = model.to(device)

    dataset_FBP = DicomTestDataset(glob(dataset_A + "/*.dcm"),
                                   glob(dataset_B + "/*.dcm"),
                                   transforms.Compose(transform_list))
    test_loader = DataLoader(dataset_FBP, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    model.train()
    for i, target in enumerate(test_loader):
        # get the inputs
        data, label = target["data"].to(device), target["label"]  # .to(device)
        data_path, label_path = target["data_path"][0], target["label_path"][0]
        input_data = data.clone()

        # compute output
        output = model(data)

        dir_name = os.path.basename(os.path.dirname(data_path))
        file_name = os.path.basename(data_path)
        tensor2dicom(output + input_data, data_path, f"{save_path}/{dir_name}/{file_name}")

        #break

    print('Test Finished')


if __name__ == "__main__":

    load_modelAtoB_path = "./models/AtoB_210/last_TC.pth"
    load_modelBtoA_path = "./models/BtoA_210/last_TC.pth"
    dataset_A_list = glob("E:/LiverCT/20210901-20211130_sort/testA/*")
    dataset_B_list = glob("E:/LiverCT/20210901-20211130_sort/testB/*")
    saveAtoB_path = "./results/AtoB_test"
    saveBtoA_path = "./results/BtoA_test"

    for target_idx in range(len(dataset_A_list)):
        test(load_modelAtoB_path, dataset_A_list[target_idx], dataset_B_list[target_idx], saveAtoB_path)
        test(load_modelBtoA_path, dataset_B_list[target_idx], dataset_A_list[target_idx], saveBtoA_path)