import os
import random
import glob
import argparse
from PIL import Image
#  python RandomCrop.py D:/礁灰岩/图像/Coarse_OTSU/ D:\礁灰岩\crop512 512 5 true
parser = argparse.ArgumentParser()
parser.add_argument("input_folder", type=str, default='D:/礁灰岩/图像/Coarse_OTSU/', help="root dir of img dataset")
parser.add_argument("output_folder", type=str, default='out', help="out dir to save the generated image")
parser.add_argument("crop_size", type=int, default=256, help="size of image after cropped ")
parser.add_argument("num_crops", type=int, default=10, help="num of crops")
parser.add_argument("rotate", type=bool, default=False, help="rotate the croped images")

opt = parser.parse_args()
print(opt)

def batch_random_crop_images(input_folder, output_folder, crop_size, num_crops,rotate=False):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中所有图像文件的路径
    image_files = glob.glob(os.path.join(input_folder, "*.bmp"))  

    for image_file in image_files:
        image = Image.open(image_file)
        image_name = os.path.splitext(os.path.basename(image_file))[0]  # 提取图像文件名（不包含扩展名）

        for i in range(num_crops):
            x = random.randint(0, image.width - crop_size)
            y = random.randint(0, image.height - crop_size)
            cropped_image = image.crop((x, y, x + crop_size, y + crop_size))

            # 生成保存路径，并保存裁剪后的图像
            save_path = os.path.join(output_folder, f"{image_name}_crop{i+1}.jpg")
            cropped_image.save(save_path)
            if rotate:
                rotated_image_90 = cropped_image.rotate(90)
                rotated_image_180 = cropped_image.rotate(180)
                rotated_image_270 = cropped_image.rotate(270)

                # 生成旋转后的保存路径，并保存旋转后的图像
                save_path_90 = os.path.join(output_folder, f"{image_name}_crop{i+1}_rotate90.jpg")
                save_path_180 = os.path.join(output_folder, f"{image_name}_crop{i+1}_rotate180.jpg")
                save_path_270 = os.path.join(output_folder, f"{image_name}_crop{i+1}_rotate270.jpg")

                rotated_image_90.save(save_path_90)
                rotated_image_180.save(save_path_180)
                rotated_image_270.save(save_path_270)
            
        print(f"Saved all the cropped image of {image_file} to {save_path}") 

# 示例用法
input_folder = opt.input_folder
output_folder = opt.output_folder
crop_size = opt.crop_size
num_crops = opt.num_crops
rotate = opt.rotate

batch_random_crop_images(input_folder, output_folder, crop_size, num_crops, rotate)