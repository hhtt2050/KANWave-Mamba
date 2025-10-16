import argparse
import logging
import os
import random
import shutil
import sys
import time
from medpy.metric.binary import dc, hd95, asd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from thop import profile
from .networks.vision_mamba_KAN import KANWave_Mamba as KWM
from .config import get_config
from .dataloaders import utils
from .dataloaders.dataset_RICE import VOCAgriculture
from .utils import losses, metrics, ramps
from .metrics import compute_mIoU, compute_dice, compute_pixel_accuracy, compute_precision, compute_recall

parser = argparse.ArgumentParser()
#more details will be released after researchreceived.
args = parser.parse_args()
config = get_config(args)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    device = torch.device("cuda:0")
    model_config = config.clone()
    model = KWM(model_config,
                    img_size=args.patch_size,
                    num_classes=args.num_classes,
                    in_chans=3
                    ).to(device)
    model.load_from(config)

    # Parameters and FLOPs calculation
    input = torch.randn(1, 3, *args.patch_size).to(device)  # [1,3,H,W]
    flops, params = profile(model, inputs=(input,))
    logging.info(f"== Number of model parameters: {params / 1e6:.2f}M ==")
    logging.info(f"== Theoretical calculations: {flops / 1e9:.2f}GFLOPs ==")

    db_train = VOCAgriculture(
        root=os.path.join(args.root_path, args.disease),
        split='train',
        transform='train',
        crop_size=args.patch_size
    )
    db_val = VOCAgriculture(
        root=os.path.join(args.root_path, args.disease),
        split='val',
        transform='val',
        crop_size=args.patch_size
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                sample_idx = 0
                image = volume_batch[sample_idx, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)

                outputs_argmax = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_argmax[sample_idx, ...] * 50, iter_num)

                labs = label_batch[sample_idx, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num % 200 == 0:
                model.eval()
                all_preds = []
                all_labels = []

                with torch.no_grad():
                    for val_batch in valloader:
                        image = val_batch['image'].to(device)
                        label = val_batch['label'].to(device).long()

                        outputs = model(image)
                        preds = torch.argmax(outputs, dim=1)  # [B, H, W]
                        all_preds.append(preds.detach().cpu())
                        all_labels.append(label.detach().cpu())

                # Combine all samples
                all_preds = torch.cat(all_preds, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

                # Calculation of global indicators
                avg_miou = compute_mIoU(all_preds, all_labels, num_classes=2)
                avg_dice = compute_dice(all_preds, all_labels,num_classes=2).item()
                #                avg_acc = compute_pixel_accuracy(all_preds, all_labels).item()
                precision = compute_precision(all_preds, all_labels)
                recall = compute_recall(all_preds, all_labels)

                # Logging to TensorBoard and Logs
                writer.add_scalar('val/mIoU', avg_miou, iter_num)
                writer.add_scalar('val/Dice', avg_dice, iter_num)
                #                writer.add_scalar('val/Accuracy', avg_acc, iter_num)
                writer.add_scalar('val/Precision', precision, iter_num)
                writer.add_scalar('val/Recall', recall, iter_num)

                logging.info(f'\n=== Validation @ Iter {iter_num} ===')
                logging.info(f'  mIoU     : {avg_miou:.4f}')
                logging.info(f'  Dice     : {avg_dice:.4f}')
                #                logging.info(f'  Accuracy : {avg_acc:.4f}')
                logging.info(f'  Recall   : {recall:.4f}')
                logging.info(f'  Precision: {precision:.4f}')

                # Preservation of best models (based on Dice)
                if avg_dice > best_performance:
                    best_performance = avg_dice
                    save_mode_path = os.path.join(snapshot_path, f'iter_{iter_num}_dice_{avg_dice:.4f}.pth')
                    save_best = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    logging.info(f'Saved new best model: Dice={avg_dice:.4f}')

                model.train()

            if iter_num % 4000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = os.path.join(
        "../model",
        f"{args.exp}_{args.disease}",
        args.model
    )
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

