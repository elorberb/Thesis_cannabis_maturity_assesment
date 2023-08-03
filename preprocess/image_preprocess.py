from image_manipulation import *
import os


def extract_file_name(file_path):
    """
  Extracts the file name from a file path.

  Args:
    file_path (str): The file path.

  Returns:
    str: The file name without the extension.
  """
    # Split the file path into a list of strings
    parts = file_path.split("/")
    # Get the last element in the list (the file name)
    file_name = parts[-1]
    # Split the file name into a list of strings
    parts = file_name.split(".")
    # Get the first element in the list (the file name without the extension)
    file_name = parts[0]

    return file_name


def read_images_and_names(dir_path, func=None):
    """
    Read all images and their corresponding names in a directory and return them as a list of tuples.
    Each tuple in the list contains a NumPy array representing the image and a string representing the name of the image.

    Parameters:
        dir_path (str): The path to the directory containing the images and their names.

    Returns:
        images_and_names: A list of tuples, where each tuple contains a NumPy array representing the image and a string representing the name of the image.
        :param func: Function to apply the image if mentioned
    """
    images_and_names = []
    for filename in os.listdir(dir_path):
        # Check if file is an image
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".PNG") or filename.endswith(
                ".JPG") or filename.endswith(".jpeg"):
            # Read image and store as NumPy array
            file_path = os.path.join(dir_path, filename)
            image = cv2.imread(file_path)
            if func is not None:
                image = func(image)
            image_name = extract_file_name(file_path)
            images_and_names.append((image, image_name))
    return images_and_names


def cut_images(image, patch_height=300, patch_width=300):
    patches = []
    patches_with_cords = []
    width, height, _ = image.shape
    for i in range(0, height, patch_height):
        for j in range(0, width, patch_width):
            patch = image[j:j + patch_width, i:i + patch_height]
            patches.append(patch)
            patches_with_cords.append((patch, i, j))
    return np.array(patches), np.array(patches_with_cords)


def save_patches(image_name, patches, dir_path):
    # Create the image name subdirectory if it does not exist
    subdir_path = os.path.join(dir_path, image_name)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)

    for i, patch in enumerate(patches):
        # Use the image name subdirectory as the file path
        file_path = os.path.join(subdir_path, f"{image_name}_p{i}.jpg")
        cv2.imwrite(file_path, patch)


def apply_function_to_images(images_and_names, func):
    modified_images_and_names = []
    for image, name in images_and_names:
        modified_image = func(image)
        modified_images_and_names.append((modified_image, name))
    return modified_images_and_names


def preprocess_patches(dir_path, trichomes_images):
    """
    Preprocesses the patches of trichome images to remove monochromatic and blurry patches.

    Parameters:
    - dir_path (str): The path to the directory where the preprocessed patches will be saved.
    - trichomes_images (list): A list of tuples, where each tuple consists of a trichome image and the image name.

    Returns:
    - None
    """
    # Preprocess the patches of each image
    for (image, image_name) in trichomes_images:
        # Cut the image into patches
        patches, _ = cut_images(image)

        # Calculate the sharpness and monochromatic values for each patch
        sharpness_values = [calculate_sharpness(patch) for patch in patches]

        # Get the average sharpness and monochromatic values for the patches
        avg_sharpness = np.mean(sharpness_values)

        # Preprocess each patch
        preprocessed_patches = [patch for patch, sharpness in zip(patches, sharpness_values) if sharpness > avg_sharpness]

        # Save the preprocessed patches to the specified directory
        save_patches(image_name, preprocessed_patches, dir_path)
