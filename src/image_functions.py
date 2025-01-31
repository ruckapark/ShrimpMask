import cv2
import numpy as np
from pathlib import Path

def sort_image_dataset(directory: Path, inputs_extension=".jpg", outputs_extension=".jpg"):
    """
    Prepares input-output pairs for a deep learning dataset from a given directory.

    This function checks the directory for input files with a specific extension and matches them
    to corresponding output files based on naming and extension. It raises errors if insufficient
    files are found or if there are mismatches.

    Args:
        directory (Path): Path object or path string pointing to the input directory.
        inputs_extension (str): Extension of input files (default: ".jpg").
        outputs_extension (str): Extension of output files (default: ".jpg").

    Returns:
        tuple: A tuple containing two lists - inputs and outputs, both as Path objects.

    Raises:
        ValueError: If the directory does not contain enough input files or mismatched files.
    """
    # Ensure `directory` is a Path object
    if not isinstance(directory, Path):
        directory = Path(directory)
    
    # Check if the directory exists and is a directory
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Provided directory '{directory}' does not exist or is not a directory.")

    # Gather all potential input files
    raw_inputs = [f for f in directory.iterdir() if f.suffix == inputs_extension]
    if len(raw_inputs) < 2:
        raise ValueError(f"Not enough input files with extension '{inputs_extension}' in the directory.")

    # Create lists for inputs and outputs
    inputs, outputs = [], []
    for f in raw_inputs:
        output = f.with_suffix(outputs_extension)
        if output.exists() and output.is_file():
            inputs.append(f)
            outputs.append(output)

    # Check if sufficient input-output pairs are found
    if len(outputs) < 2:
        raise ValueError("Not enough valid input-output pairs in the directory.")

    return inputs, outputs

def load_image_to_array(image_name: str, mask = False) -> np.ndarray:
    """
    Ouvre une image depuis le dossier `data/` et la stocke dans un array NumPy.

    Args:
        image_name (str): Le nom de l'image (par exemple, "image.jpg").

    Returns:
        np.ndarray: L'image sous forme de tableau NumPy.
    """
    # Chemin vers le dossier data/
    data_folder = Path(__file__).parent.parent / "data"
    
    # Chemin complet de l'image
    image_path = data_folder / image_name
    
    # Vérifie si le fichier existe
    if not image_path.exists():
        raise FileNotFoundError(f"L'image '{image_name}' n'existe pas dans {data_folder}.")
    
    # Charge l'image avec OpenCV
    image_array = cv2.imread(str(image_path))  # Utilise str() pour convertir Path en string compatible

    #white non black pixels if mask
    if mask:
        non_black_mask = np.any(image_array > 0, axis=-1)
        image_array[non_black_mask] = [255, 255, 255]
    
    # Vérifie si l'image a été chargée correctement
    if image_array is None:
        raise ValueError(f"Impossible de charger l'image '{image_name}'. Assurez-vous qu'il s'agit d'un fichier image valide.")
    
    return image_array

def compress_image(image: np.ndarray, dim=224, grayscale=True, mask = False):
    """
    Compress and resize an image, optionally converting it to grayscale, and save it to the specified output path.

    Args:
        image (np.ndarray): The input image array to process.
        output (Path): The file path where the processed image will be saved.
        extension (str): File extension for the saved image (default is '.jpg').
        dim (int): Target dimension for both width and height (default is 384x384).
        grayscale (bool): If True, converts the image to grayscale before resizing.

    Returns:
        np.ndarray: The processed image array.
    """
    # Convert the image to an array format (ensure compatibility with OpenCV operations)
    image_array = load_image_to_array(image, mask = mask)

    # Dilate the image if specified
    if mask:
        # Define the kernel size for dilation (e.g., 3x3)
        kernel_size = 10
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        image_array = cv2.dilate(image_array, kernel, iterations=1)
    
    # Resize the image to the specified dimensions (224*224 by default)
    if grayscale:
        image_compress = cv2.resize(cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY),(dim, dim),interpolation=cv2.INTER_LANCZOS4)
    else:
        # Resize without changing color channels
        image_compress = cv2.resize(image_array,(dim, dim),interpolation=cv2.INTER_LANCZOS4)
    
    # Return compressed image
    return image_compress

# Exemple d'utilisation
if __name__ == "__main__":

    #Obtrain type of extension functions
    mask_extensions = [f".ov{i}" for i in range(1,10)] #bitmap (BMP) file extension names
    extension = mask_extensions[0]

    try:
        root = Path(__file__).parents[1]
        print(root)
        image_name = "0.5 D R1 F1.jpg"  # Remplacez par le nom de votre image

        image_path = root / "data_test" / image_name
        mask_path = image_path.with_suffix(extension)
        found_mask = False

        # Check if the file exists with the current or other extensions
        if not mask_path.exists():
            for ext in mask_extensions:
                mask_path = image_path.with_suffix(ext)
                if mask_path.exists():
                    found_mask = True
                    extension = ext
                    break
        else:
            found_mask = True

        if found_mask:
            print(f"Mask found at: {mask_path}")
        else:
            print(f"No mask file found for: {image_path}")
            
        #read image file
        image_array = load_image_to_array(image_path)
        print(f"L'image {image_path.name} a été chargée avec succès. Dimensions : {image_array.shape}")

        #read mask image
        mask_array = load_image_to_array(mask_path)
        
        non_black_mask = np.any(mask_array > 0, axis=-1)
        mask_array[non_black_mask] = [255, 255, 255]
        print(f"L'image {mask_path.name} a été chargée avec succès. Dimensions : {mask_array.shape}")

        #resize to 224 and greyscale
        image_compress = cv2.resize(cv2.cvtColor(image_array , cv2.COLOR_BGR2GRAY),(224,224), interpolation = cv2.INTER_LANCZOS4)
        mask_compress = cv2.resize(cv2.cvtColor(mask_array , cv2.COLOR_BGR2GRAY),(224,224), interpolation = cv2.INTER_LANCZOS4)
        
        cv2.imshow('mask2',mask_compress) # sufficient quality

        print(f"L'image compressée {mask_path.name} a été chargée avec succès. Dimensions : {mask_compress.shape}")

        #cv2 won't write back to ov BMP format
        cv2.imwrite(image_path.with_suffix(".compressed.jpg"),image_compress)
        cv2.imwrite(image_path.with_suffix(".compressed_mask.jpg"),mask_compress)

    except Exception as e:
        print(f"Erreur : {e}")