import torch
import os
import sys
from scipy import stats 
#from dataloader.AGIQA1K import load_image_label
from dataloader.AGIQA3K import load_image_label
from dataloader.AIGCIQA2023 import load_image, load_label, load_prompt
from config import get_parser
from util import get_logger, log_and_print
import random
import torch.backends.cudnn as cudnn
from transformers import AutoProcessor, AutoModel


sys.path.append('../')
torch.backends.cudnn.enabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

def calc_probs(processor, model, prompt, images):
    
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
    
        return scores

if __name__ == '__main__':

    args = get_parser().parse_known_args()[0]

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    base_logger = get_logger(f'exp/TIER-4K.log', args.log_info)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''label_path = "./Dataset/AGIQA-1k/AIGC_MOS_Zscore.xlsx"
    image_path = "./Dataset/AGIQA-1k/file"
    generated_image, label, text_prompt = load_image_label(image_path, label_path)'''

    label_path = "./Dataset/AGIQA-3K/data.csv"
    image_path = "./Dataset/AGIQA-3K/image"
    generated_image, label, text_prompt = load_image_label(image_path, label_path)

    '''root_path = './Dataset/AIGCIQA2023/Image/allimg'
    label_path = './Dataset/AIGCIQA2023/DATA/MOS/mosz1.mat'
    prompt_path = './Dataset/AIGCIQA2023/prompt.xlsx'
    generated_image = load_image(root_path)
    label = load_label(label_path)
    text_prompt = load_prompt(prompt_path)'''
    
    device = "cuda"
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

    pred_scores = []
    with torch.no_grad():
        for index in range(len(generated_image)):
            preds = calc_probs(processor, model, text_prompt[index], generated_image[index])
            pred_scores.extend([i.item() for i in preds])

    rho_s, _ = stats.spearmanr(pred_scores, label)
    rho_p, _ = stats.pearsonr(pred_scores, label)

    log_and_print(base_logger, f'spearmanr_correlation: {rho_s}, pearsonr_correlation: {rho_p}')


    # run: HF_ENDPOINT=https://hf-mirror.com python Pick_evalaute.py

    



