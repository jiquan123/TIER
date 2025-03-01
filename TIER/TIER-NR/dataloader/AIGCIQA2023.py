import torch
import torchvision.transforms as tr
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
from PIL import Image
import scipy.io
import pandas as pd
import numpy as np

def random_shuffle(generated_image, label, text_prompt):
    randnum = 1
    np.random.seed(randnum)
    np.random.shuffle(generated_image)
    np.random.seed(randnum)
    np.random.shuffle(label)
    np.random.seed(randnum)
    np.random.shuffle(text_prompt)
    return generated_image, label, text_prompt



# AIGCIQA-2023
def load_image(root_path):
    # 
    image_files = [f for f in os.listdir(root_path) if f.endswith('.jpg') or f.endswith('.png')]
    image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))  # AIGCIQA-2023

    image_list = []
    for name in image_files:
        file = os.path.join(root_path, name)
        image = Image.open(file).convert('RGB')
        image_list.append(image)
    return image_list


def load_label(path):
    mat_data = scipy.io.loadmat(path)
    label = mat_data['MOSz']
    label_list = []
    for i in range(len(label)):
        label_list.append(label[i][0])
    return label_list

def load_prompt(path):
    data = pd.read_excel(path)
    text_prompt = data['prompt']
    text_prompt_list = []
    for prompt in text_prompt:
        text_prompt_list.append(prompt)
    return text_prompt_list


class AIGCIQA2023Dataset(Dataset):
    def __init__(self, image, label, text_prompt, transforms, text_encoder_path):
        self.image = image
        self.label = label
        self.text_prompt = text_prompt
        self.transforms = transforms
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        data = {}
        data['image'] = self.transforms(self.image[idx])
        data['MOS_score'] = self.label[idx]
        text_prompt = self.text_prompt[idx]
        encoded = self.tokenizer.encode_plus(
            text_prompt,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512
        )
        data['prompt'], data['attention_mask'] = encoded['input_ids'], encoded['attention_mask']
        return data


def get_AIGCIQA2023q_dataloaders(args):
    root_path = './Dataset/AIGCIQA2023/Image/allimg'
    label_path = './Dataset/AIGCIQA2023/DATA/MOS/mosz1.mat'
    prompt_path = './Dataset/AIGCIQA2023/prompt.xlsx'
    text_encoder_path = "./bert-base-uncased"
    image = load_image(root_path)
    label = load_label(label_path)
    text_prompt = load_prompt(prompt_path)

    image, label, text_prompt = random_shuffle(image, label, text_prompt)
    percent = int(len(image) * 0.8)
    train_image = image[:percent]
    train_label = label[:percent]
    train_text_prompt = text_prompt[:percent]
    test_image = image[percent:]
    test_label = label[percent:]
    test_text_prompt = text_prompt[percent:]

    if args.backbone == 'inceptionv4':
        resize_img_size = 320
        crop_img_size = 299
    else:
        resize_img_size = 256
        crop_img_size = 224

    train_transforms = tr.Compose([
        tr.Resize(resize_img_size),
        tr.RandomCrop(crop_img_size),
        tr.RandomHorizontalFlip(),
        tr.ToTensor(),
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transforms = tr.Compose([
        tr.Resize(resize_img_size),
        tr.CenterCrop(crop_img_size),
        tr.ToTensor(),
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(
        AIGCIQA2023Dataset(train_image, train_label, train_text_prompt, train_transforms, text_encoder_path),
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True)

    dataloaders['test'] = torch.utils.data.DataLoader(
        AIGCIQA2023Dataset(test_image, test_label, test_text_prompt, test_transforms, text_encoder_path),
        batch_size=args.test_batch_size,
        shuffle=False,
        pin_memory=True)
    return dataloaders

def get_AIGCIQA2023a_dataloaders(args):
    root_path = './Dataset/AIGCIQA2023/Image/allimg'
    label_path = './Dataset/AIGCIQA2023/DATA/MOS/mosz2.mat'
    prompt_path = './Dataset/AIGCIQA2023/prompt.xlsx'
    text_encoder_path = "./bert-base-uncased"
    image = load_image(root_path)
    label = load_label(label_path)
    text_prompt = load_prompt(prompt_path)

    image, label, text_prompt = random_shuffle(image, label, text_prompt)
    percent = int(len(image) * 0.8)
    train_image = image[:percent]
    train_label = label[:percent]
    train_text_prompt = text_prompt[:percent]
    test_image = image[percent:]
    test_label = label[percent:]
    test_text_prompt = text_prompt[percent:]

    if args.backbone == 'inceptionv4':
        resize_img_size = 320
        crop_img_size = 299
    else:
        resize_img_size = 256
        crop_img_size = 224

    train_transforms = tr.Compose([
        tr.Resize(resize_img_size),
        tr.CenterCrop(crop_img_size),
        tr.RandomHorizontalFlip(),
        tr.ToTensor(),
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transforms = tr.Compose([
        tr.Resize(resize_img_size),
        tr.CenterCrop(crop_img_size),
        tr.ToTensor(),
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(
        AIGCIQA2023Dataset(train_image, train_label, train_text_prompt, train_transforms, text_encoder_path),
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True)

    dataloaders['test'] = torch.utils.data.DataLoader(
        AIGCIQA2023Dataset(test_image, test_label, test_text_prompt, test_transforms, text_encoder_path),
        batch_size=args.test_batch_size,
        shuffle=False,
        pin_memory=True)
    return dataloaders



def get_AIGCIQA2023c_dataloaders(args):
    root_path = './Dataset/AIGCIQA2023/Image/allimg'
    label_path = './Dataset/AIGCIQA2023/DATA/MOS/mosz3.mat'
    prompt_path = './Dataset/AIGCIQA2023/prompt.xlsx'
    text_encoder_path = "./bert-base-uncased"
    image = load_image(root_path)
    label = load_label(label_path)
    text_prompt = load_prompt(prompt_path)

    image, label, text_prompt = random_shuffle(image, label, text_prompt)
    percent = int(len(image) * 0.8)
    train_image = image[:percent]
    train_label = label[:percent]
    train_text_prompt = text_prompt[:percent]
    test_image = image[percent:]
    test_label = label[percent:]
    test_text_prompt = text_prompt[percent:]

    if args.backbone == 'inceptionv4':
        resize_img_size = 320
        crop_img_size = 299
    else:
        resize_img_size = 256
        crop_img_size = 224

    train_transforms = tr.Compose([
        tr.Resize(resize_img_size),
        tr.CenterCrop(crop_img_size),
        tr.RandomHorizontalFlip(),
        tr.ToTensor(),
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transforms = tr.Compose([
        tr.Resize(resize_img_size),
        tr.CenterCrop(crop_img_size),
        tr.ToTensor(),
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(
        AIGCIQA2023Dataset(train_image, train_label, train_text_prompt, train_transforms, text_encoder_path),
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True)

    dataloaders['test'] = torch.utils.data.DataLoader(
        AIGCIQA2023Dataset(test_image, test_label, test_text_prompt, test_transforms, text_encoder_path),
        batch_size=args.test_batch_size,
        shuffle=False,
        pin_memory=True)
    return dataloaders

