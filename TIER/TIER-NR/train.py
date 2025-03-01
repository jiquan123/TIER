import torch
import torch.nn as nn
import os
import sys
from scipy import stats
from tqdm import tqdm
from backbone.resnet import resnet18_backbone, resnet50_backbone
from backbone.inceptionv4 import inceptionv4
from backbone.vit import ViTExtractor, SwinExtractor
from dataloader.AGIQA1K import get_AGIQA1K_dataloaders
from dataloader.AGIQA3K import get_AGIQA3K_dataloaders
from dataloader.AIGCIQA2023 import get_AIGCIQA2023q_dataloaders, get_AIGCIQA2023a_dataloaders, get_AIGCIQA2023c_dataloaders
from model import Encoder, MLP
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

    base_logger = get_logger(f'exp/TIER.log', args.log_info)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.backbone == 'resnet50':
        backbone = resnet50_backbone().to(device)
        encoder = Encoder(backbone).to(device)
        regressor = MLP(2048).to(device)
    elif args.backbone == 'resnet18':
        backbone = resnet18_backbone().to(device)
        encoder = Encoder(backbone).to(device)
        regressor = MLP(512).to(device)
    elif args.backbone == 'vit':
        backbone = ViTExtractor().to(device)
        encoder = Encoder(backbone).to(device)
        regressor = MLP(1024).to(device)
    elif args.backbone == 'swin':
        backbone = SwinExtractor().to(device)
        encoder = Encoder(backbone).to(device)
        regressor = MLP(1024).to(device)
    else:
        backbone = inceptionv4(num_classes=1000, pretrained='imagenet').to(device)
        encoder = Encoder(backbone).to(device)
        regressor = MLP(1536).to(device)

    if args.benchmark == 'AIGCIQA2023q':
        dataloaders = get_AIGCIQA2023q_dataloaders(args)
    elif args.benchmark == 'AIGCIQA2023a':
        dataloaders = get_AIGCIQA2023a_dataloaders(args)
    elif args.benchmark == 'AIGCIQA2023c':
        dataloaders = get_AIGCIQA2023c_dataloaders(args)
    elif args.benchmark == 'AGIQA3K':
        dataloaders = get_AGIQA3K_dataloaders(args)
    else:
        dataloaders = get_AGIQA1K_dataloaders(args)

    criterion = nn.MSELoss(reduction='mean').cuda()
    # criterion = nn.SmoothL1Loss(reduction='mean')
    optimizer = torch.optim.Adam([*encoder.parameters()] + [*regressor.parameters()],
                                 lr=args.lr, weight_decay=args.weight_decay)

    epoch_best = 0
    rho_s_best = 0.0
    rho_p_best = 0.0
    for epoch in range(args.num_epochs):
        log_and_print(base_logger, f'Epoch: {epoch}')

        for split in ['train', 'test']:
            true_scores = []
            pred_scores = []

            if split == 'train':
                encoder.train()
                regressor.train()
                torch.set_grad_enabled(True)
            else:
                encoder.eval()
                regressor.eval()
                torch.set_grad_enabled(False)

            for data in tqdm(dataloaders[split]):
                true_scores.extend(data['MOS_score'].numpy())

                image = data['image'].to(device)  # B, C, H, W
                text_prompt = data['prompt'].to(device)
                mask = data['attention_mask'].to(device)

                feature = encoder(image, text_prompt, mask)
                preds = regressor(feature).view(-1)

                pred_scores.extend([i.item() for i in preds])

                if split == 'train':
                    loss = criterion(preds, data['MOS_score'].float().to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            rho_s, _ = stats.spearmanr(pred_scores, true_scores)
            rho_p, _ = stats.pearsonr(pred_scores, true_scores)

            log_and_print(base_logger, f'{split} spearmanr_correlation: {rho_s}, pearsonr_correlation: {rho_p}')

        if rho_s > rho_s_best:
            rho_s_best = rho_s
            epoch_best = epoch
            log_and_print(base_logger, '##### New best spearmanr correlation #####')
            # path = 'ckpts/' + str(rho) + '.pt'
            path = 'ckpts/' + 'best_model.pt'
            torch.save({'epoch': epoch,
                        'encoder': encoder.state_dict(),
                        'regressor': regressor.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'rho_best': rho_s_best}, path)
        if rho_p > rho_p_best:
            rho_p_best = rho_p
            log_and_print(base_logger, '##### New best pearsonr correlation #####')
        log_and_print(base_logger, ' EPOCH_best: %d, SRCC_best: %.6f, PLCC_best: %.6f' % (epoch_best, rho_s_best, rho_p_best))

