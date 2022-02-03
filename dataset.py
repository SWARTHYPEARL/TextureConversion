
import torch
from util import open_dicom
import torch.utils.data as data

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
    def __init__(self, data_list, label_list, transform = None):
        super(DicomPairDataset, self).__init__()
        self.data_list = data_list
        self.label_list = label_list
        self.transform = transform

    def __getitem__(self, index):
        data = open_dicom(self.data_list[index], True)
        label = open_dicom(self.label_list[index], True)
        if self.transform is not None:
            data = self.transform(data)
            label = self.transform(label)
        return {"data": data, "label": label}

    def __len__(self):
        return len(self.data_list)


class DicomTestDataset(data.Dataset):
    def __init__(self, data_list, label_list, transform = None):
        super(DicomTestDataset, self).__init__()
        self.data_list = data_list
        self.label_list = label_list
        self.transform = transform

    def __getitem__(self, index):
        data = open_dicom(self.data_list[index], True)
        label = open_dicom(self.label_list[index], True)
        if self.transform is not None:
            data = self.transform(data)
            label = self.transform(label)
        return {"data": data, "label": label, "data_path": self.data_list[index], "label_path": self.label_list[index]}

    def __len__(self):
        return len(self.data_list)