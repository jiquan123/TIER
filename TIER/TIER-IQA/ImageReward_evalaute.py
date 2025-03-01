import torch
import os
import sys
from scipy import stats
import ImageReward 
#from dataloader.AGIQA1K import load_image_label
from dataloader.AGIQA3K import load_image_label
from dataloader.AIGCIQA2023 import load_image, load_label, load_prompt
from config import get_parser
from util import get_logger, log_and_print
import random
import torch.backends.cudnn as cudnn

sys.path.append('../')
torch.backends.cudnn.enabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
if __name__ == '__main__':

    args = get_parser().parse_known_args()[0]

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    base_logger = get_logger(f'exp/TIER-ImageReward.log', args.log_info)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''label_path = "./Dataset/AGIQA-1k/AIGC_MOS_Zscore.xlsx"
    image_path = "./Dataset/AGIQA-1k/file"
    generated_image, label, text_prompt = load_image_label(image_path, label_path)'''

    '''label_path = "./Dataset/AGIQA-3K/data.csv"
    image_path = "./Dataset/AGIQA-3K/image"
    generated_image, label, text_prompt = load_image_label(image_path, label_path)'''

    root_path = './Dataset/AIGCIQA2023/Image/allimg'
    label_path = './Dataset/AIGCIQA2023/DATA/MOS/mosz3.mat'
    prompt_path = './Dataset/AIGCIQA2023/prompt.xlsx'
    generated_image = load_image(root_path)
    label = load_label(label_path)
    text_prompt = load_prompt(prompt_path)

    
    model = ImageReward.load("ImageReward-v1.0")
    pred_scores = []
    with torch.no_grad():
        for index in range(len(generated_image)):
            preds = model.score(text_prompt[index], generated_image[index])
            pred_scores.append(preds)

    rho_s, _ = stats.spearmanr(pred_scores, label)
    rho_p, _ = stats.pearsonr(pred_scores, label)

    log_and_print(base_logger, f'spearmanr_correlation: {rho_s}, pearsonr_correlation: {rho_p}')

    # run: HF_ENDPOINT=https://hf-mirror.com python ImageReward_evalaute.py

    # pip install image-reward


    



