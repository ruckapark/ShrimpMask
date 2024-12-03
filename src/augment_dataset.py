import cv2
import numpy as np
from pathlib import Path

if __name__ == "__main__":

    #Pass directory of raw data files
    root_dir = Path(__file__).parents[1]
    data_dir = root_dir / "data"
    augmented_data_dir = root_dir / "augmented_data"

    data_files = [f for f in data_dir.iterdir() if "input" in str(f)]
    no_files = len(data_files)

    counter = 0
    for i in range(no_files):

        reverse,flip = 0,0

        while not reverse or flip:
            reverse = np.random.randint(0,2)
            flip = np.random.randint(0,2)

        input_im = cv2.imread(data_dir / f"input{i}.jpg", cv2.IMREAD_GRAYSCALE)
        output_im = cv2.imread(data_dir / f"output{i}.jpg", cv2.IMREAD_GRAYSCALE)

        input_im_aug = input_im.copy()
        output_im_aug = output_im.copy()

        if reverse:
            input_im_aug = np.flip(input_im_aug,0)
            output_im_aug = np.flip(output_im_aug,0)

        if flip:
            input_im_aug = np.flip(input_im_aug,1)
            output_im_aug = np.flip(output_im_aug,1)

        cv2.imwrite(str(augmented_data_dir / f"input{counter}.jpg"), input_im)
        cv2.imwrite(str(augmented_data_dir / f"output{counter}.jpg"), output_im)

        counter += 1

        cv2.imwrite(str(augmented_data_dir / f"input{counter}.jpg"), input_im_aug)
        cv2.imwrite(str(augmented_data_dir / f"output{counter}.jpg"), output_im_aug)

        counter += 1