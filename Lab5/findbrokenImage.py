import os
from PIL import Image
import PIL

def clean_corrupted_images(folder_paths):

    corrupted_files = []
    
    for folder_path in folder_paths:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with Image.open(file_path) as img:
                    img.verify()
                    
            except (PIL.UnidentifiedImageError, IOError, SyntaxError) as e:
                print(f"Corrupted image found: {filename} - Error: {e}")
                corrupted_files.append(file_path)
                
                os.remove(file_path)
                print(f"Removed: {filename}")
    
    print(f"\nCleaning complete. Removed {len(corrupted_files)} corrupted files.")
    return corrupted_files

# Define your dataset folders
folder_paths = [
    'custom_dataset/train/class_0',
    'custom_dataset/train/class_1', 
    'custom_dataset/train/class_2',
    'custom_dataset/val/class_0',
    'custom_dataset/val/class_1',
    'custom_dataset/val/class_2'
]

# Run the cleaning script
corrupted = clean_corrupted_images(folder_paths)