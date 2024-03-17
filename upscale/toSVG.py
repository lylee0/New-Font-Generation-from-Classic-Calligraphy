import aspose.words as aw
import os

def find_png_files(folder_path):
    png_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            file_path = os.path.join(folder_path, file_name)
            png_files.append(file_path)
    return png_files

# Specify the path to the folder you want to search
folder_path = "./upscaled_images"

png_files = find_png_files(folder_path)

doc = aw.Document()
builder = aw.DocumentBuilder(doc)

save_path = "./upscaled_svg"

if not os.path.exists(save_path):
    # Create the folder
    os.makedirs(save_path)

for i in range(len(png_files)):
    file = png_files[i]
    shape = builder.insert_image(file)

    character_name = os.path.splitext(os.path.basename(png_files[i]))[0]
    file_name = "{}.svg".format(character_name)
    output_file = os.path.join(save_path, file_name)
    shape.get_shape_renderer().save(output_file, aw.saving.ImageSaveOptions(aw.SaveFormat.SVG))
