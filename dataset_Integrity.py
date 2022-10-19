
import os
import shutil
from glob import glob

if __name__ == "__main__":

    target_dir = r"C:\Users\SNUBH\SP_work\Python_Project\TextureConversion\dataset\20200727-20201231_2-5mm"

    series_list = ["DELAY FBP_", "DELAY_", "L-A FBP_", "L-A_", "PRE FBP_", "PRE_", "PVP FBP_", "PVP_"]

    for target_patient in glob(target_dir + "/*"):
        for series_idx, target_series in enumerate(glob(target_patient + "/*")):
            if series_list[series_idx] not in os.path.basename(target_series):
                print(f"Not match: {series_list[series_idx]} - {target_series}")


