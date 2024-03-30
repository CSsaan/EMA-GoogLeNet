import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class AIM500Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir, 'original')
        self.label_folder = os.path.join(root_dir, 'mask')
        self.image_filenames = os.listdir(self.image_folder)
        self.label_filenames = os.listdir(self.label_folder)
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Normalize((0.5,), (0.5,))  # 如果你使用的是灰度图像，这里应该是 (0.5,)
            transforms.Resize((224, 224)),  # 如果你需要调整图像大小
            # transforms.RandomHorizontalFlip(),  # 如果你需要随机水平翻转
            # transforms.RandomRotation(10),  # 如果你需要随机旋转
            # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 如果你需要随机仿射变换
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 如果你需要颜色抖动
            # transforms.RandomGrayscale(p=0.2),  # 如果你需要随机灰度
            # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # 如果你需要随机透视变换
            # transforms.RandomCrop(224, padding=4),  # 如果你需要随机裁剪
            # transforms.RandomResizedCrop(224),  # 如果你需要随机调整大小和裁剪
            # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),  # 如果你需要随机擦除
            # transforms.RandomSizedCrop(224),  # 如果你需要随机大小裁剪
        ])
        self.transform_label = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Resize((256, 256)),
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_filenames[idx])
        label_name = os.path.join(self.label_folder, self.label_filenames[idx])
        image = Image.open(img_name).convert('RGB')
        label = Image.open(label_name).convert('L')  # Assuming label is grayscale
        image = self.transform_image(image)
        label = self.transform_label(label)
        return image, label

if __name__ == '__main__':
    # 设定训练数据集路径
    train_dataset = AIM500Dataset(root_dir='/workspace/EMA-GoogLeNet/data/AIM500')
    # 设置批量大小和是否随机打乱数据
    batch_size = 16
    shuffle = True
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    # 检查数据加载是否正常工作
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        # 这里你可以在模型上训练，images 和 labels 分别是图像和相应的标签数据
