import pandas as pd
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets


class PandasConverter:
    def __init__(self, file_path, label_column, features_column, type='excel'):
        """
        :param file_path: Excel的文件路径
        :param label_column: 标签列的名称
        :param features_column: 特征列的名称
        :param type: 文件格式(excel、csv)
        """
        self.file_path = file_path
        self.label_column = label_column
        self.features_column = features_column
        self.type = type

    def convert(self):
        """
        :return: (特征张量，标签张量)
        """
        if self.type == 'excel':
            df = pd.read_excel(self.file_path)
        elif self.type == 'csv':
            df = pd.read_csv(self.file_path)
        else:
            raise "Please Input Excel or Csv.Use 'excel' or 'csv' to join"
        labels = df[self.label_column].values
        features = df[self.features_column].values
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return features_tensor, labels_tensor


# # 示例使用
# if __name__ == "__main__":
#     # 初始化Excel数据转换器
#     excel_converter = PandasConverter("D:\\Math_model\\2024_xiaosai\\B\\fujian\\excel\\data.xlsx", label_column='中国通胀率',features_column=['美国通胀率','美国GDP'])
#
#     # 转换数据
#     excel_features, excel_labels = excel_converter.convert()
#     print(f'Excel特征张量形状: {excel_features.shape}')
#     print(f'Excel标签张量形状: {excel_labels.shape}')


class ImageConverter:
    def __init__(self, image_size=(28, 28), transform=True):
        """
        初始化图像数据转换器
        :param image_size: 图像大小，默认为(28, 28)
        :param transform: 是否需要对图片进行特征工程
        """
        self.image_size = image_size
        self.transform = transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def convert_to_tensor(self, image_paths, labels):
        """
        将图像数据转换为tensor数据
        :param image_paths: 图像文件路径列表
        :param labels: 标签列表
        :return: (特征张量, 标签张量)
        """
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('L')  # 转为灰度图
            if self.transform:
                image = self.transform(image)
            images.append(image)

        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return images_tensor.view(images_tensor.size(0), -1), labels_tensor


# 示例使用
# if __name__ == "__main__":
#     # 初始化图像数据转换器
#     image_converter = ImageConverter(image_size=(28, 28))
#
#     # 图像路径和标签
#     image_paths = ["C:\\Users\\lenovo\\Desktop\\85755c8b728d4428a72dda4a3accc57d.jpg"]
#     image_labels = [0]
#
#     # 转换数据
#     image_features, image_labels = image_converter.convert_to_tensor(image_paths, image_labels)
#     print(f'图像特征张量形状: {image_features.shape}')
#     print(f'图像标签张量形状: {image_labels.shape}')

"""
后续还需要添加对于图片数据集的预处理
"""


class BatchDataLoader:
    def __init__(self, dataset, batch_size=64, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    @staticmethod
    def load_mnist(data_dir='./data', batch_size=64):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)
        train_loader = BatchDataLoader(train_dataset, batch_size=batch_size, shuffle=True).get_loader()
        test_loader = BatchDataLoader(test_dataset, batch_size=batch_size, shuffle=False).get_loader()
        return train_loader, test_loader


def RandomDataLoader(num_samples, num_features, num_classes, batch_size, test_size=0.2, with_test_set=False):
    """
    生成随机特征和标签，并转换为 DataLoader。

    参数：
    num_samples : 样本数量。
    num_features : 特征数量。
    num_classes : 类别数量。
    batch_size : 批大小。
    test_size : 测试集大小比例，默认为 0.2。
    with_test_set : 是否需要测试集，默认为 False。

    返回：
    train_loader : 训练集 DataLoader 对象。
    test_loader : 测试集 DataLoader 对象（如果 with_test_set=True）。
    """
    # 生成随机特征和标签
    features = torch.randn(num_samples, num_features)
    labels = torch.randint(0, num_classes, (num_samples,))

    # 划分训练集和测试集
    if with_test_set:
        features_train, features_test, labels_train, labels_test = train_test_split(
            features, labels, test_size=test_size, random_state=42)
        # 转换为 TensorDataset
        train_dataset = TensorDataset(features_train, labels_train)
        test_dataset = TensorDataset(features_test, labels_test)

        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
    else:
        # 转换为 TensorDataset
        dataset = TensorDataset(features, labels)

        # 创建 DataLoader
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return train_loader

