
import os
import pydicom
import numpy as np

def resize_normalize(image):
    image = np.array(image, dtype=np.float64)
    image -= -1024 # np.min(image)
    image /= 4095 # np.max(image)
    return image

def open_dicom(path, normalize = False):
    """
    load dicom pixel data and return HU value
    Args:
        path: dicom file path

    Returns: dicom HU pixel array
    """
    image_medical = pydicom.dcmread(path)
    image_data = image_medical.pixel_array

    hu_image = image_data * image_medical.RescaleSlope + image_medical.RescaleIntercept
    hu_image[hu_image < -1024] = -1024
    hu_image[hu_image > 3071] = 3071

    #image_window = window_image(image_hu.copy(), window_level, window_width)

    hu_image = np.expand_dims(hu_image, axis=2)  # (512, 512, 1)
    if normalize:
        hu_image = resize_normalize(hu_image)

    return hu_image  # use single-channel

def tensor2dicom(input_image, original_path, save_path):
    image_numpy = input_image.squeeze(0).cpu().detach().numpy()
    # apply normalization boundary
    image_numpy[image_numpy < 0.0] = 0.0
    image_numpy[image_numpy > 1.0] = 1.0

    original_dicom = pydicom.dcmread(original_path)
    original_numpy = original_dicom.pixel_array
    #print(original_dicom.file_meta.TransferSyntaxUID)

    #original_dicom.PixelData = np.round((image_numpy + 1) / 2.0 * (np.max(original_numpy * original_dicom.RescaleSlope))).astype(np.uint16).tobytes()
    #original_dicom.PixelData = np.round((image_numpy + 1) / 2.0 * 4095).astype(np.uint16).tobytes()
    original_dicom.PixelData = np.round(image_numpy * 4095).astype(np.uint16).tobytes()
    original_dicom.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    #original_dicom.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    if not os.path.isdir(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    original_dicom.save_as(save_path)