
from glob import glob
import os

if __name__ == "__main__":

    data_path = "E:/LiverCT/20210901-20211130_sort/testA"
    label_path = "E:/LiverCT/20210901-20211130_sort/testB"

    data_path_list = glob(data_path + "/*/*.dcm")
    label_path_list = glob(label_path + "/*/*.dcm")

    for i, data in enumerate(data_path_list):
        label = label_path_list[i]

        if data.replace("testA", "testB") == label:
            continue
        else:
            print(f"data: {data}")
            print(f"labe: {label}")

