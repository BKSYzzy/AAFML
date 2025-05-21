import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
import random
import collections

class CIFARFS(Dataset):
    """
    CIFAR-FS：
    root/
        |- train/
            |- class1/
                |- img1.png
                |- img2.png ...
            |- class2/ ...
        |- test/
            |- class5/
            |- class6/ ...
        |- train.txt 
        |- test.txt
    """

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize=32):
        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.setsz = self.n_way * self.k_shot
        self.querysz = self.n_way * self.k_query
        self.resize = resize
        self.mode = mode

        if mode == 'train':
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        self.class_names = self.load_txt(os.path.join(root, f"{mode}.txt"))
        self.img_paths = collections.defaultdict(list)
        self.label_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

        data_dir = os.path.join(root, mode)
        for cls in self.class_names:
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.exists(cls_dir):
                raise FileNotFoundError(f"Category folder {cls_dir} does not exist")
            for img_name in os.listdir(cls_dir):
                self.img_paths[cls].append(os.path.join(cls_dir, img_name))

        self.cls_num = len(self.class_names)
        self.create_batch(batchsz)

    def load_txt(self, txt_path):
        with open(txt_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names

    def create_batch(self, batchsz):
        self.support_x_batch = []
        self.query_x_batch = []
        for _ in range(batchsz):
            if len(self.class_names) < self.n_way:
                raise ValueError(f"Insufficient number of classes: need {self.n_way} class, but only {len(self.class_names)} class available")
            selected_classes = [self.class_names[0]] + random.sample(self.class_names[1:], self.n_way - 1)

            support_x, query_x = [], []
            for cls in selected_classes:
                all_imgs = self.img_paths[cls]
                if len(all_imgs) < self.k_shot + self.k_query:
                    raise ValueError(f"Insufficient sample size for category {cls}")
                selected = random.sample(all_imgs, self.k_shot + self.k_query)
                support_x.append(selected[:self.k_shot])
                query_x.append(selected[self.k_shot:])

            self.support_x_batch.append(support_x)
            self.query_x_batch.append(query_x)

    def __getitem__(self, index):
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        support_y = np.zeros(self.setsz, dtype=np.int32)
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        query_y = np.zeros(self.querysz, dtype=np.int32)

        flatten_support = [item for sublist in self.support_x_batch[index] for item in sublist]
        support_y = [self.label_to_idx[os.path.basename(os.path.dirname(p))] for p in flatten_support]

        flatten_query = [item for sublist in self.query_x_batch[index] for item in sublist]
        query_y = [self.label_to_idx[os.path.basename(os.path.dirname(p))] for p in flatten_query]

        unique = np.unique(support_y)
        support_y_rel = np.zeros_like(support_y)
        query_y_rel = np.zeros_like(query_y)
        for idx, lbl in enumerate(unique):
            support_y_rel[support_y == lbl] = idx
            query_y_rel[query_y == lbl] = idx

        for i, path in enumerate(flatten_support):
            support_x[i] = self.transform(path)
        for i, path in enumerate(flatten_query):
            query_x[i] = self.transform(path)

        return support_x, torch.LongTensor(support_y_rel), query_x, torch.LongTensor(query_y_rel)

    def __len__(self):
        return self.batchsz



if __name__ == '__main__':
    dataset = CIFARFS(
        root='cifarfs',
        mode='train',
        batchsz=100,
        n_way=5,
        k_shot=5,
        k_query=15,
        resize=32
    )
    support, s_y, query, q_y = dataset[0]
    print("Support set shape:", support.shape)
    print("Query set shape:", query.shape)
