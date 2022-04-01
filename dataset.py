
import torch
from util import open_dicom
import torch.utils.data as data

import numpy as np
from glob import glob
from torchvision import transforms

class DicomToTensor(object):
    """
    convert dicom numpy to normalized tensor(torch)
    """

    def __call__(self, target_dicom):
        # numpy HWD
        # torch DHW
        target_dicom = target_dicom.transpose((2, 0, 1))
        return torch.from_numpy(target_dicom).type(torch.FloatTensor)#.type(torch.DoubleTensor)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class DicomPairDataset(data.Dataset):
    def __init__(self, data_list, label_list, transform = None, input_channel = 1):
        super(DicomPairDataset, self).__init__()
        self.data_list = data_list
        self.label_list = label_list
        self.transform = transform
        self.input_channel = input_channel

    def __getitem__(self, index):
        if self.input_channel == 1:
            data = open_dicom(self.data_list[index], True)
        elif self.input_channel == 3:
            data_front = open_dicom(self.data_list[index - 1 if index > 0 else 0], True)
            data_center = open_dicom(self.data_list[index], True)
            data_rear = open_dicom(self.data_list[index + 1 if index < self.__len__() - 1 else self.__len__() - 1], True)
            data = np.concatenate((data_front, data_center, data_rear), axis=2)
        else:
            raise Exception("input channel dose not equal either 1 or 3 in dataset process.")

        label = open_dicom(self.label_list[index], True)
        if self.transform is not None:
            data = self.transform(data)
            label = self.transform(label)
        return {"data": data, "label": label}

    def __len__(self):
        return len(self.data_list)


class DicomTestDataset(data.Dataset):
    def __init__(self, data_list, label_list, transform = None, input_channel = 1):
        super(DicomTestDataset, self).__init__()
        self.data_list = data_list
        self.label_list = label_list
        self.transform = transform
        self.input_channel = input_channel

    def __getitem__(self, index):
        if self.input_channel == 1:
            data = open_dicom(self.data_list[index], True)
        elif self.input_channel == 3:
            data_front = open_dicom(self.data_list[index - 1 if index > 0 else 0], True)
            data_center = open_dicom(self.data_list[index], True)
            data_rear = open_dicom(self.data_list[index + 1 if index < self.__len__() - 1 else self.__len__() - 1],
                                   True)
            data = np.concatenate((data_front, data_center, data_rear), axis=2)
        else:
            raise Exception("input channel dose not equal either 1 or 3 in dataset process.")

        label = open_dicom(self.label_list[index], True)
        if self.transform is not None:
            data = self.transform(data)
            label = self.transform(label)
        return {"data": data, "label": label, "data_path": self.data_list[index], "label_path": self.label_list[index]}

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":

    dataA_path = r"D:\LiverCT\20200724-20200729_train\DELAY\dataA"
    dataB_path = r"D:\LiverCT\20200724-20200729_train\DELAY\dataB"

    transform_list = [
        DicomToTensor()
    ]

    dataset_AtoB = DicomPairDataset(glob(dataA_path + "/**/*.dcm", recursive=True),
                                    glob(dataB_path + "/**/*.dcm", recursive=True),
                                    transforms.Compose(transform_list),
                                    input_channel=3)

    print(dataset_AtoB[0]["data"].shape)
    print(dataset_AtoB[0]["label"].shape)