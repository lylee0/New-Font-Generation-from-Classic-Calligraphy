#!/usr/bin/python
# -*- coding: UTF-8 -*-

from super_image import ImageLoader, DrlnModel
from PIL import Image
import os

def find_png_files(folder_path):
    png_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            file_path = os.path.join(folder_path, file_name)
            png_files.append(file_path)
    return png_files

# Specify the path to the folder you want to search
folder_path = "./images"

# Call the function to find PNG files in the folder
png_files = find_png_files(folder_path)
'''print(png_files)
# Open the file in write mode
with open('output.txt', 'w', encoding='utf-8') as f:
    # Iterate over the elements of the list
    for item in png_files:
        item = os.path.basename(item)
        # Write each element to a new line in the file
        f.write('\"' + item + '\"' + ', ')
f.close()'''

model = DrlnModel.from_pretrained('eugenesiow/drln', scale=4)

for i in range(len(png_files)):

    image = Image.open(png_files[i])
    inputs = ImageLoader.load_image(image)
    preds = model(inputs)
    preds = model(preds)
    

    # Specify the output folder
    output_folder = './upscaled_images'

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    character_name = os.path.splitext(os.path.basename(png_files[i]))[0]
    #file_name = "{}_upscaled.png".format(character_name)
    file_name = "{}.png".format(i)
    # Join the output folder and encoded filename
    output_file = os.path.join(output_folder, file_name)

    ImageLoader.save_image(preds, output_file)

    new_path = os.path.join(output_folder, "{}_upscaled.png".format(character_name))

    os.rename(output_file, new_path)

    num = i + 1
    if num == 1:
        print("{} image created".format(num))
    else:
        print("{} images created".format(num))
