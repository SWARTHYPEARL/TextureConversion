
import os
import shutil
from glob import glob
from dataset import DicomTestDataset, DicomToTensor
from model import TextureConversion
from util import tensor2dicom

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def test(load_model_path: str, dataset_A: str, dataset_B: str, save_path: str, input_channel: int):

    print(f"load_model: {load_model_path}")
    print(f"dataset_A: {dataset_A}")
    print(f"dataset_B: {dataset_B}")
    print(f"save_path: {save_path}")

    transform_list = [
        DicomToTensor()
    ]

    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    model = TextureConversion(input_channel)  # .type(torch.DoubleTensor)
    model.load_state_dict(torch.load(load_model_path, map_location=device))
    model = model.to(device)

    dataset_AtoB = DicomTestDataset(glob(dataset_A + "/*.dcm"),
                                    glob(dataset_B + "/*.dcm"),
                                    transforms.Compose(transform_list),
                                    input_channel)
    test_loader = DataLoader(dataset_AtoB, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    model.eval()
    for i, target in enumerate(test_loader):
        # get the inputs
        data, label = target["data"].to(device), target["label"]  # .to(device)
        data_path, label_path = target["data_path"][0], target["label_path"][0]
        if input_channel == 1:
            input_data = data.clone()
        elif input_channel == 3:
            input_data = torch.index_select(data, 1, torch.tensor([1]).to(device))
        else:
            raise Exception("input channel dose not equal either 1 or 3 in train process.")

        # compute output
        output = model(data)

        dir_name = os.path.basename(os.path.dirname(data_path))
        file_name = os.path.basename(data_path)
        tensor2dicom(output + input_data, data_path, f"{save_path}/conversion_of_{dir_name}/{file_name}")

        #break
    #shutil.copytree(dataset_A, f"{save_path}/original_{os.path.basename(dataset_A)}")

    print("Test Finished\n")


def LiverCT_test():

    # load_modelAtoB_path = "./models/AtoB_210/last_TC.pth"
    # load_modelBtoA_path = "./models/BtoA_210/last_TC.pth"
    # dataset_A_list = glob(r"C:\Users\SNUBH\SP_work\Python_Project\TextureConversion\dataset\20210101-20220131_2-5mm\*\*FBP*")
    # dataset_B_list = glob("E:/LiverCT/20210901-20211130_sort/testB/*")
    # saveAtoB_path = "./results/AtoB_test"
    # saveBtoA_path = "./results/BtoA_test"

    load_model_path = "./models/20200724-20200729_train"
    #load_model_list = ["300_TC.pth", "200_TC.pth", "100_TC.pth"]
    load_model_list = load_model_list = [os.path.basename(x) for x in glob(load_model_path + f"/PVP/AtoB/*")] # ["050_TC.pth"]
    save_path = r"Y:\SP_work\billion\ubuntu\Python_Project\TextureConversion\results\20200724-20200729_train_PVP_validation"
    input_channel = 1

    load_modelAtoB_path = []
    load_modelBtoA_path = []
    dataset_A_list = []
    dataset_B_list = []
    saveAtoB_path = []
    saveBtoA_path = []
    phase_list = ["DELAY", "L-A", "PVP"]
    target_dir = "./dataset/total_20200727-20220131_validation"
    for target_model in load_model_list:
        for target_patient in glob(target_dir + "/*"):
            for target_series in glob(target_patient + "/*"):
                if not os.path.isdir(target_series):
                    continue

                target_recon = os.path.basename(target_series)
                target_phase = None
                if "FBP" in target_recon:
                    target_phase = target_recon.split(" ")[0]
                    if target_phase in phase_list:
                        dataset_A_list.append(target_series)
                        saveAtoB_path.append(save_path + f"/{target_model}/{os.path.basename(target_patient)}")
                        #load_modelAtoB_path.append(load_model_path + f"/{target_phase}/AtoB/{target_model}")
                        load_modelAtoB_path.append(load_model_path + f"/PVP/AtoB/{target_model}")
                else:
                    target_phase = target_recon.split("_")[0]
                    if target_phase in phase_list:
                        dataset_B_list.append(target_series)
                        saveBtoA_path.append(save_path + f"/{target_model}/{os.path.basename(target_patient)}")
                        #load_modelBtoA_path.append(load_model_path + f"/{target_phase}/BtoA/{target_model}")
                        load_modelBtoA_path.append(load_model_path + f"/PVP/BtoA/{target_model}")

    # for target_idx in range(len(dataset_A_list)):
    #    if not (dataset_A_list[target_idx][:109] == dataset_B_list[target_idx][:109]):
    #        print(dataset_A_list[target_idx])
    # exit(0)

    for target_idx in range(len(dataset_A_list)):
        test(load_modelAtoB_path[target_idx], dataset_A_list[target_idx], dataset_B_list[target_idx],
             saveAtoB_path[target_idx], input_channel)
        test(load_modelBtoA_path[target_idx], dataset_B_list[target_idx], dataset_A_list[target_idx],
             saveBtoA_path[target_idx], input_channel)

        # break


if __name__ == "__main__":

    LiverCT_test()
