from PIL import Image
import os

def find_png_files(folder_path):
    png_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            file_path = os.path.join(folder_path, file_name)
            png_files.append(file_path)
    return png_files

folder_path = "./upscaled_images"

png_files = find_png_files(folder_path)

output_folder = './background_removed'

if not os.path.exists(output_folder):
    # Create the folder
    os.makedirs(output_folder)

for i in range(len(png_files)):

    image = Image.open(png_files[i])

    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    threshold = 50  # Adjust this value according to your image

    width, height = image.size
    pixels = image.load()

    for x in range(width):
        for y in range(height):
            r, g, b, a = pixels[x, y]

            if r > threshold and g > threshold and b > threshold:
                pixels[x, y] = (r, g, b, 0)  # Set the pixel to transparent
                a = 0
                
            if a != 0:
                pixels[x, y] = (0, 0, 0, a)

    character_name = os.path.splitext(os.path.basename(png_files[i]))[0][0]
    file_name = "{}.png".format(i)
    output_file = os.path.join(output_folder, file_name)

    #output_path = 'output.png'
    image.save(output_file)

    new_path = os.path.join(output_folder, "{}_removed.png".format(character_name))

    os.rename(output_file, new_path)

    num = i + 1
    if num == 1:
        print("{} image created".format(num))
    else:
        print("{} images created".format(num))
