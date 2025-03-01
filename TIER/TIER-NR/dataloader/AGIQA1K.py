import torch
import torchvision.transforms as tr
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
from PIL import Image
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

# AGIQA-1k

def load_image_label(image_path, label_path):
    data = pd.read_excel(label_path)
    labels = data['MOS']
    image_files_name = data['Image']
    text_prompt = data['Prompt']

    image_list = []
    for name in image_files_name:
        file = os.path.join(image_path, str('{}'.format(name)))
        image = Image.open(file).convert('RGB')
        image_list.append(image)
    label_list = []
    for label in labels:
        label_list.append(label)
    text_prompt_list = []
    for prompt in text_prompt:
        text_prompt_list.append(prompt)
    return image_list, label_list, text_prompt_list


class AGIQA1kDataset(Dataset):
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


def get_AGIQA1K_dataloaders(args):
    label_path = "./Dataset/AGIQA-1k/AIGC_MOS_Zscore.xlsx"
    image_path = "./Dataset/AGIQA-1k/file"
    text_encoder_path = "./bert-base-uncased"
    image, label, text_prompt = load_image_label(image_path, label_path)
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

    dataloaders['train'] = torch.utils.data.DataLoader(AGIQA1kDataset(train_image, train_label, train_text_prompt, train_transforms, text_encoder_path),
                                                       batch_size=args.train_batch_size,
                                                       shuffle=True,
                                                       pin_memory=True)

    dataloaders['test'] = torch.utils.data.DataLoader(AGIQA1kDataset(test_image, test_label, test_text_prompt, test_transforms, text_encoder_path),
                                                      batch_size=args.test_batch_size,
                                                      shuffle=False,
                                                      pin_memory=True)
    return dataloaders



'''from config import get_parser

args = get_parser().parse_known_args()[0]
dataloaders = get_AGIQA1K_dataloaders(args)

for data in tqdm(dataloaders['train']):
    image = data['image']
    prompt = data['prompt']
    print(prompt.shape)'''