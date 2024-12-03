import cv2
from pathlib import Path
import image_functions as image

if __name__ == "__main__":

    #Pass directory of raw data files
    root_dir = Path(__file__).parents[1]
    data_dir = root_dir / "data"
    rawdata_dir = Path(r"I:\Dataset_Gammarus")

    #Locate input and output files
    input_extension = '.jpg'
    output_extension = '.ov1'

    inputs,outputs = image.sort_image_dataset(rawdata_dir,input_extension,output_extension)

    print(f"Inputs begins with {inputs[0].name}, and contains {len(inputs)} files\n")
    print(f"Outputs begins with {outputs[0].name}, and contains {len(outputs)} files\n")

    #Convert images to greyscale black and white 384*384 pixel images
    for i in range(len(inputs)):

        #file save names
        input_ = image.compress_image(inputs[i])
        output_ = image.compress_image(outputs[i])

        #output file save names
        f_input = data_dir / f"input{i}.jpg"
        f_output = data_dir / f"output{i}.jpg"

        cv2.imwrite(str(f_input),input_)
        cv2.imwrite(str(f_output),output_)