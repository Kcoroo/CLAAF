import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
import scipy.io
import glob
import cv2
import math 
import os
import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import AutoTokenizer, AutoImageProcessor
from torchvision import transforms

class SorcenDatasets(Dataset):
    def __init__(self, phase):
        # self.tokenizer = AutoTokenizer.from_pretrained('./models/roberta-base')
        self.tokenizer = AutoTokenizer.from_pretrained('./models/clip-vit-base')
        self.processor = AutoImageProcessor.from_pretrained('./models/resnet-50')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.load_datasets()
        
        if phase == 'train':
            self.img_data, _, _, _ = train_test_split(self.img_data, self.label, test_size=0.3, random_state=42)
            self.text_data, _, self.label, _ = train_test_split(self.text_data, self.label, test_size=0.3, random_state=42)
        elif phase == 'valid':
            _, self.img_data, _, _ = train_test_split(self.img_data, self.label, test_size=0.3, random_state=42)
            _, self.text_data, _, self.label = train_test_split(self.text_data, self.label, test_size=0.3, random_state=42)
            self.img_data, _, _, _ = train_test_split(self.img_data, self.label, test_size=0.5, random_state=42)
            self.text_data, _, self.label, _ = train_test_split(self.text_data, self.label, test_size=0.5, random_state=42)
        else:
            _, self.img_data, _, _ = train_test_split(self.img_data, self.label, test_size=0.3, random_state=42)
            _, self.text_data, _, self.label = train_test_split(self.text_data, self.label, test_size=0.3, random_state=42)
            _, self.img_data, _, _ = train_test_split(self.img_data, self.label, test_size=0.5, random_state=42)
            _, self.text_data, _, self.label = train_test_split(self.text_data, self.label, test_size=0.5, random_state=42)
        

    def load_datasets(self):
        positive_data = pd.read_csv("./data/positive.csv")
        negative_data = pd.read_csv("./data/negative.csv")
        
        img_datas = []
        text_datas = []
        labels = []
        for i, row in positive_data.iterrows():
            file_tail = ['jpg', 'png', 'jpeg', 'gif', 'JPG']
            file_tail = [tail for tail in file_tail if os.path.exists(os.path.join('./data/images/positive', str(row['序号']) + f'.{tail}'))]
            img_name = os.path.join('./data/images/positive', str(row['序号']) + f'.{file_tail[0]}')
            image = Image.open(img_name)
            # image = self.transform(image)
            content = row['内容']
            label = 0
            img_datas.append(image)
            text_datas.append(content)
            labels.append(label)
        for i, row in negative_data.iterrows():
            file_tail = ['jpg', 'png', 'jpeg', 'gif', 'JPG']
            file_tail = [tail for tail in file_tail if os.path.exists(os.path.join('./data/images/negative', str(row['序号']) + f'.{tail}'))]
            img_name = os.path.join('./data/images/negative', str(row['序号']) + f'.{file_tail[0]}')
            image = Image.open(img_name)
            # image = self.transform(image)
            content = row['内容']
            label = 1
            img_datas.append(image)
            text_datas.append(content)
            labels.append(label)
        
        self.img_data = img_datas
        self.text_data = text_datas
        self.label = labels

    def __getitem__(self, index):
        text_str = self.text_data[index]
        data = self.tokenizer.encode_plus(text_str,
            padding='max_length',   # 一律补0到max_length长度
            max_length=128,
            return_token_type_ids=True,
            truncation=True, 
            return_tensors='pt'
        )   # 返回length，标识长度
        input_ids = data['input_ids'][0]    # input_ids:编码之后的数字
        attention_mask = data['attention_mask'][0]     # attention_mask:补零的位置是0,其他位置是1
        token_type_ids = data['token_type_ids'][0]   # 第一个句子和特殊符号的位置是0，第二个句子的位置是1(包括第二个句子后的[SEP])
        
        image = self.img_data[index]
        image_inputs = self.processor(image.convert('RGB'), return_tensors="pt")['pixel_values'][0]

        label = torch.tensor(self.label[index], dtype=torch.long)

        return input_ids, attention_mask, token_type_ids, image_inputs, label
    

    def __len__(self):
        return len(self.label)

if __name__ == "__main__":
    train_data = SorcenDatasets('train')
    # print(train_data)
    print([i.shape for i in next(iter(train_data))])