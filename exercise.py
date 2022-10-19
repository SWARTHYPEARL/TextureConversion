
from glob import glob
import os

if __name__ == "__main__":

    target_dir = "./dataset/total_20200727-20220131_validation"

    for target_recon in glob(target_dir + "/*/*"):

        os.rename(target_recon, os.path.dirname(target_recon) + f"/{os.path.basename(target_recon).split('original_')[1]}")
