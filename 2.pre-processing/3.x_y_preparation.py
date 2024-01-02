from dicom_to_image import DicomToImage
from nrrd_to_image import NrrdToImage


def generate(index, size, dicom_file, nrrd_file):
    path_prefix = f"{size[0]}"

    DicomToImage(
        dicom_file_path=dicom_file,
        output_path=f'{path_prefix}_data/x/x_{index}.png',
        standard_size=size,
    ).process()

    NrrdToImage(
        nrrd_file_path=nrrd_file,
        output_path=f"{path_prefix}_data/y/y_{index}.png",
        size=size,
    ).process()


index = 14
first_path = "MyDataset/"
dcm_file = "FO-6157774497763423242"
generate(
    index=index,
    size=(512, 512),
    dicom_file=f"{first_path}{index}/{dcm_file}.dcm",
    nrrd_file=f"{first_path}{index}/Segmentation.nrrd"
)
