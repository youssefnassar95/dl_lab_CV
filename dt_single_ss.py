import os
import random
import numpy as np
import argparse
import torch
import time
from torchvision.transforms import ToTensor
from utils.meters import AverageValueMeter
from utils.weights import load_from_weights
from utils import check_dir, set_random_seed, accuracy, mIoU, get_logger
from models.att_segmentation import AttSegmentator
from torch.utils.tensorboard import SummaryWriter
from data.transforms import get_transforms_binary_segmentation
from models.pretraining_backbone import ResNet18Backbone
from data.segmentation import DataReaderSingleClassSemanticSegmentationVector, DataReaderSemanticSegmentationVector

set_random_seed(0)
global_step = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str,
                        help="folder containing the data")
    parser.add_argument('--pretrained_model_path', type=str, default='')
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--bs', type=int, default=16, help='batch_size')
    parser.add_argument('--att', type=str, default='sdotprod',
                        help='Type of attention. Choose from {additive, cosine, dotprod, sdotprod}')
    parser.add_argument('--size', type=int, default=256, help='image size')
    parser.add_argument('--snapshot-freq', type=int,
                        default=5, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="")
    args = parser.parse_args()

    hparam_keys = ["lr", "bs", "att"]
    args.exp_name = "_".join(
        ["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    try:
        args.jobid = os.environ["PBS_JOBID"].replace(
            ".lmbtorque.informatik.uni-freiburg.de", "")
    except KeyError:
        args.jobid = "local"

    args.exp_name += "_{}_{}".format(args.exp_suffix, args.jobid)

    args.output_folder = check_dir(os.path.join(
        args.output_root, 'dt_attseg', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.output_folder, args.exp_name)
    img_size = (args.size, args.size)

    # model
    pretrained_model = ResNet18Backbone(True)
    # TODO: Complete the documentation for AttSegmentator model
    # raise NotImplementedError("TODO: Build model AttSegmentator model")
    model = AttSegmentator(5, pretrained_model.features,
                           att_type=args.att, img_size=img_size).cuda()
    if os.path.isfile(args.pretrained_model_path):
        model = load_from_weights(model, args.pretrained_model_path, logger)

    if os.path.isfile(args.pretrained_model_path):
        model = load_from_weights(model, args.pretrained_model_path, logger)

    # dataset
    data_root = args.data_folder
    train_transform, val_transform, train_transform_mask, val_transform_mask = get_transforms_binary_segmentation(
        args)
    vec_transform = ToTensor()
    train_data = DataReaderSingleClassSemanticSegmentationVector(
        os.path.join(data_root, "imgs/train2014"),
        os.path.join(data_root, "aggregated_annotations_train_5classes.json"),
        transform=train_transform,
        vec_transform=vec_transform,
        target_transform=train_transform_mask,
        flip=True, crop=True, crop_size=args.size,
    )
    # Note that the dataloaders are different.
    # During validation we want to pass all the semantic classes for each image
    # to evaluate the performance.
    val_data = DataReaderSemanticSegmentationVector(
        os.path.join(data_root, "imgs/val2014"),
        os.path.join(data_root, "aggregated_annotations_val_5classes.json"),
        transform=val_transform,
        vec_transform=vec_transform,
        target_transform=val_transform_mask
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True,
                                               num_workers=3, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False,
                                             num_workers=3, pin_memory=True, drop_last=False)

    # TODO: loss
    criterion = torch.nn.CrossEntropyLoss()
    # TODO: SGD optimizer (see pretraining)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # raise NotImplementedError("TODO: loss function and SGD optimizer")

    log = SummaryWriter(args.logs_folder)
    expdata = "  \n".join(["{} = {}".format(k, v)
                           for k, v in vars(args).items()])
    log.add_text("experiment_details", expdata, 0)
    logger.info(expdata)
    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    best_val_miou = 0.0
    for epoch in range(15):
        logger.info("Epoch {}".format(epoch))
        train(train_loader, model, criterion, optimizer, log, logger)
        val_results = validate(
            val_loader, model, criterion, log, logger, epoch)

        # save model
        val_loss = val_results["loss"]
        val_iou = val_results["miou"]
        save = (epoch % args.snapshot_freq == 0 and epoch >
                0 and val_loss < best_val_loss)
        save_as_best = best_val_miou < val_iou and epoch > 0
        if save:
            save_model(model, optimizer, args, epoch,
                       val_loss, val_iou, logger)
            best_val_loss = val_loss
        if save_as_best:
            save_model(model, optimizer, args, epoch,
                       val_loss, val_iou, logger, best=True)
            best_val_miou = val_iou


def train(loader, model, criterion, optimizer, log, logger):
    logger.info("Training")
    global global_step
    model.train()

    loss_meter = AverageValueMeter()
    iou_meter = AverageValueMeter()
    time_meter = AverageValueMeter()
    steps_per_epoch = len(loader.dataset) / loader.batch_size

    start_time = time.time()
    batch_time = time.time()
    for idx, (img, v_class, label) in enumerate(loader):
        img = img.cuda()
        v_class = v_class.float().cuda().squeeze()
        logits, alphas = model(img, v_class, out_att=True)
        logits = logits.squeeze()
        labels = (torch.nn.functional.interpolate(label.cuda(),
                                                  size=logits.shape[-2:]).squeeze(1)*256).long()
        loss = criterion(logits, labels)
        iou = mIoU(logits, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.add(loss.item())
        iou_meter.add(iou)
        time_meter.add(time.time()-batch_time)

        if idx % 50 == 0:
            text_print = "Epoch {:.4f} Avg loss = {:.4f} mIoU = {:.4f} Time {:.2f} (Total:{:.2f}) Progress {}/{}".format(
                global_step / steps_per_epoch, loss_meter.mean, iou_meter.mean, time_meter.mean, time.time()-start_time, idx, int(steps_per_epoch))
            logger.info(text_print)
            scalar_dict = {"loss": loss_meter.mean, "miou": iou_meter.mean,
                           "epoch_prog": idx / (len(loader.dataset) / loader.batch_size)}
            save_in_log(log, global_step, set="train", scalar_dict=scalar_dict)
            loss_meter.reset()
            iou_meter.reset()

        if idx == 0 or idx == int(steps_per_epoch/2):
            image_dict = {'sample': img, 'vec': v_class,
                          'gt': labels, 'pred': logits, 'att': alphas}
            save_in_log(log, global_step, set="train", image_dict=image_dict)

        global_step += 1
        batch_time = time.time()
    time_txt = "batch time: {:.2f} total time: {:.2f}".format(
        time_meter.mean, time.time()-start_time)
    logger.info(time_txt)


def validate(loader, model, criterion, log, logger, epoch=0):
    logger.info("Validating Epoch {}".format(epoch))
    model.eval()

    loss_meter = AverageValueMeter()
    iou_meter = AverageValueMeter()

    save_idx = random.sample(range(0, len(loader)-1), 3)
    save_idx = set(save_idx)

    start_time = time.time()
    for idx, (img, v_class, label) in enumerate(loader):
        img = img.squeeze(0).cuda()
        v_class = v_class.float().cuda().squeeze()
        logits, alphas = model(img, v_class, out_att=True)
        label = label.squeeze(0).unsqueeze(1)
        labels = (torch.nn.functional.interpolate(label.cuda(),
                                                  size=logits.shape[-2:]).squeeze(1)*256).long()
        loss = criterion(logits, labels)
        iou = mIoU(logits, labels)

        loss_meter.add(loss.item())
        iou_meter.add(iou)

        if idx in save_idx:
            image_dict = {'sample': img, 'vec': v_class,
                          'gt': labels, 'pred': logits, 'att': alphas}
            save_in_log(log, global_step, set="valid", image_dict=image_dict)

    results = {"loss": loss_meter.mean, "miou": iou_meter.mean}
    text_print = "Epoch {} Avg loss = {:.4f} mIoU = {:.4f} Time {:.2f}".format(
        epoch, loss_meter.mean, iou_meter.mean, time.time()-start_time)
    # , text_dict=text_dict)
    save_in_log(log, global_step, set="valid", scalar_dict=results)
    logger.info(text_print)

    return results


def save_in_log(log, save_step, set="", scalar_dict=None, text_dict=None, image_dict=None):
    if scalar_dict:
        [log.add_scalar(set+"_"+k, v, save_step)
         for k, v in scalar_dict.items()]
    if text_dict:
        [log.add_text(set+"_"+k, v, save_step) for k, v in text_dict.items()]
    if image_dict:
        for k, v in image_dict.items():
            if k == 'sample':
                log.add_images(set+"_"+k, v, save_step)
            elif k == 'vec':
                log.add_images(
                    set+"_"+k, v.unsqueeze(1).unsqueeze(1), save_step)
            elif k == 'gt':
                log.add_images(set+"_"+k, v.unsqueeze(1).expand(-1,
                                                                3, -1, -1).float()/v.max(), save_step)
            elif k == 'pred':
                log.add_images(
                    set+"_"+k, v.argmax(dim=1, keepdim=True), save_step)
            elif k == 'att':
                assert isinstance(v, list)
                for idx, alpha in enumerate(v):
                    log.add_images(
                        set+"_"+k+"_"+str(idx), (alpha.unsqueeze(1)-alpha.min())/alpha.max(), save_step)
            else:
                log.add_images(set+"_"+k, v, save_step)
    log.flush()


def save_model(model, optimizer, args, epoch, val_loss, val_iou, logger, best=False):
    # save model
    add_text_best = 'BEST' if best else ''
    logger.info('==> Saving '+add_text_best +
                ' ... epoch{} loss{:.03f} miou{:.03f} '.format(epoch, val_loss, val_iou))
    state = {
        'opt': args,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': val_loss,
        'miou': val_iou
    }
    if best:
        torch.save(state, os.path.join(args.model_folder, 'ckpt_best.pth'))
    else:
        torch.save(state, os.path.join(args.model_folder,
                                       'ckpt_epoch{}_loss{:.03f}_miou{:.03f}.pth'.format(epoch, val_loss, val_iou)))


if __name__ == '__main__':
    """
    Semantic segmentation with attention to class embeddings
    python dt_single_ss.py datasets/COCO_mini5class_medium --pretrained_model_path models/binary_segmentation.pth --exp-suffix binary_weights
    python dt_single_ss.py datasets/COCO_mini5class_medium --pretrained_model_path models/dino_weights.pth --exp-suffix dino_weights --exp-suffix dino_weights
    """
    args = parse_arguments()
    # pprint(vars(args))
    # print()
    main(args)
