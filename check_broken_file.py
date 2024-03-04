import os
from PIL import Image
from tqdm import tqdm


def check_images(root_dir):
    broken_files = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in tqdm(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                file_path = os.path.join(subdir, file)
                try:
                    img = Image.open(file_path)  # 打开图像
                    img.verify()  # 验证图像完整性
                except (IOError, SyntaxError) as e:
                    print(f'损坏的文件: {file_path}')
                    broken_files.append(file_path)

    return broken_files


# 设置你的数据集目录
dataset_directory = r'./dataset/'
broken_files = check_images(dataset_directory)

# 可选：将损坏的文件列表保存到文本文件中
with open('broken_files.txt', 'w') as file:
    for item in broken_files:
        file.write("%s\n" % item)

print(f'检查完成。发现 {len(broken_files)} 个损坏的文件。')
