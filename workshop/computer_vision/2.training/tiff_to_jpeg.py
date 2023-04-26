from PIL import Image
import os

def tiff_to_jpeg(src_dir, dst_dir):
    # Create destination directory if it doesn't exist
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Loop through all files in the source directory
    for filename in os.listdir(src_dir):
        if filename.endswith(".tif"):
            # Open the TIFF image
            with Image.open(os.path.join(src_dir, filename)) as img:
                # Convert to JPEG and save to destination directory
                img = img.convert("RGB")
                new_filename = os.path.splitext(filename)[0] + ".jpg"
                img.save(os.path.join(dst_dir, new_filename), "JPEG")
                
                
if __name__ == "__main__":
    src_dir = r"C:\Users\majmo\Downloads\Solar Panels Data\Maxar_HD_and_Native_Solar_Panel_Image_Chips\image_chips\image_chips_hd"
    dst_dir = r"C:\Users\majmo\Downloads\Solar Panels Data\Train\jpg_images"
    
    tiff_to_jpeg(src_dir, dst_dir)