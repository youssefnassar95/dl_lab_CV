import os
import numpy as np
import argparse
import torch
import torch.distributed as dist
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from pprint import pprint
from torch.utils.tensorboard import SummaryWriter

from models.pretraining_backbone import DINOHead
from models.pretraining_backbone import ResNet18Backbone

from utils.meters import AverageValueMeter
from utils import check_dir, accuracy, get_params_groups, cosine_scheduler

from data.pretraining import DataReaderPlainImg
from data.data_augmentation import TrainAugmentation, ValAugmentation

global_step = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str,
                        help="folder containing the data (crops)")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--bs', type=int, default=64, help='batch_size')
    parser.add_argument('--dataset-size', type=int, default=60000)
    parser.add_argument('--schedule', type=str, default="cosine")
    parser.add_argument('--snapshot-freq', type=int,
                        default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="")
    parser.add_argument('--weight_decay', type=float,
                        default=0.04, help='Initial value of the weight decay.')
    parser.add_argument('--weight_decay_end', type=float,
                        default=0.4, help='Final value of the weight decay.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Target LR at the end of optimization.')
    parser.add_argument('--momentum_teacher', default=0.996,
                        type=float, help='Base EMA parameter for teacher update.')
    parser.add_argument('--out_dim', default=128, type=int,
                        help='Dimensionality of the DINO head output.')
    parser.add_argument('--teacher_temp', default=0.04, type=float,
                        help='Final value of the teacher temperature.')
    parser.add_argument('--global_crops_scale', type=float, default=(0.4, 1.),
                        help='Scale range of the cropped image before resizing, relatively to the origin image.')
    parser.add_argument('--local_crops_number', type=int, default=4,
                        help='Number of small local views to generate.')
    parser.add_argument('--local_crops_scale', type=float, default=(0.05, 0.4),
                        help='Scale range of the cropped image before resizing, relatively to the origin image.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=1, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    parser.add_argument('--warmup_teacher_temp', default=0.01, type=float,
                        help='Initial value for the teacher temperature: 0.04 works well in most cases.')
    args = parser.parse_args()

    hparam_keys = ["lr", "bs"]
    args.exp_name = "_".join(
        ["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    try:
        args.jobid = os.environ["PBS_JOBID"].replace(
            ".lmbtorque.informatik.uni-freiburg.de", "")
    except KeyError:
        args.jobid = "local"

    args.exp_name += "_{}_{}".format(args.exp_suffix, args.jobid)

    args.output_folder = check_dir(
        os.path.join(args.output_root, args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # ============ building student and teacher networks ... ============
    teacher = ResNet18Backbone(pretrained=True)
    student = ResNet18Backbone(pretrained=True)

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = MultiCropWrapper(student, DINOHead(
        512, 128, norm_last_layer=True,
    ))
    teacher = MultiCropWrapper(teacher, DINOHead(
        512, 128))

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    # raise NotImplementedError("TODO: load weight initialization")

    # ============ preparing data ... ============
    train_transform = TrainAugmentation(
        args.global_crops_scale, args.local_crops_scale, args.local_crops_number)
    val_transform = ValAugmentation(
        args.local_crops_scale, args.local_crops_number)
    data_root = "/project/dl2022s/nassary/cv_assignment/data"
    train_data = DataReaderPlainImg(os.path.join(
        data_root, "train"), transform=train_transform)
    if args.dataset_size is not None:
        train_data = torch.utils.data.Subset(
            train_data, range(args.dataset_size))
    val_data = DataReaderPlainImg(os.path.join(
        data_root, "val"), transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=0,
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0,
                                             pin_memory=True, drop_last=True)

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        # total number of crops = 2 global crops + local_crops_number
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()
    # raise NotImplementedError("TODO: load weight initialization")

    # ============ preparing optimizer ... ============
    params_groups = get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)
    # optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = cosine_scheduler(
        args.lr * args.bs / 256.,
        args.min_lr,
        args.epochs, len(train_loader),
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(train_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(
        args.momentum_teacher,
        1,
        args.epochs, len(train_loader))

    log = SummaryWriter(args.logs_folder)
    expdata = "  \n".join(["{} = {}".format(k, v)
                           for k, v in vars(args).items()])
    log.add_text("experiment_details", expdata, 0)

    validate(val_loader, teacher, student, dino_loss, -1, args, log=log)

    for epoch in range(args.epochs):
        print("Epoch {}".format(epoch))
        log.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)
        train(train_loader, teacher, student, dino_loss, optimizer, lr_schedule,
              wd_schedule, momentum_schedule, epoch, fp16_scaler, log=log)
        # save model
        print('==> Saving...')
        state = {
            'opt': args,
            'epoch': epoch,
            'teacher': teacher.state_dict(),
            'student': student.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(args.model_folder,
                                       'ckpt_epoch{}.pth'.format(epoch)))
        validate(val_loader, teacher, student, dino_loss, epoch, args, log=log)


def train(data_loader, t_model, s_model, criterion, optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, log=None):
    print("Training")
    global global_step
    s_model.train()
    t_model.train()
    loss_meter = AverageValueMeter()
    for iteration, images in enumerate(data_loader):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + iteration  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # only the 2 global views pass through the teacher
            teacher_output = t_model(images[:2])
            student_output = s_model(images)
            loss = criterion(student_output, teacher_output, epoch, train=True)

        # if not math.isfinite(loss.item()):
        #     print("Loss is {}, stopping training".format(loss.item()), force=True)
        #     sys.exit(1)
        # raise NotImplementedError("TODO: load weight initialization")

        # student update
        optimizer.zero_grad()
        param_norms = None
        fp16_scaler.scale(loss).backward()
        fp16_scaler.step(optimizer)
        fp16_scaler.update()
        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(s_model.parameters(), t_model.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            # raise NotImplementedError("TODO: load weight initialization")
        loss_meter.add(loss.item())
        if iteration % 100 == 0:
            if "PBS_JOBID" not in os.environ:
                print("Avg loss = {}".format(loss_meter.mean))
            log.add_scalar("training_loss", loss_meter.mean, global_step)
            log.add_scalar("epoch_progress", iteration /
                           (len(data_loader.dataset)/data_loader.batch_size), global_step)
            loss_meter.reset()
        global_step += 1


def validate(loader, t_model, s_model, criterion, epoch, args, log=None):
    print("Validating")
    t_model.eval()
    s_model.eval()
    loss_meter = AverageValueMeter()
    feature_maps = torch.zeros(2*len(loader), 512)
    for idx, img in enumerate(loader):
        images = [im.cuda(non_blocking=True) for im in img]
        teacher_output = t_model(images[:2])
        student_output = s_model(images)
        loss = criterion(student_output, teacher_output, epoch)
        loss_meter.add(loss.item())
        feature_maps[2*idx:2 *
                     (idx+1), :] = t_model.backbone(torch.cat(images[:2], dim=0))
    print("Mean validation loss = {}".format(loss_meter.mean))
    log.add_scalar("validation_loss", loss_meter.mean, global_step)
    feature_maps_embedded = TSNE().fit_transform(feature_maps)
    # raise NotImplementedError("TODO: load weight initialization")
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=feature_maps_embedded[:, 0],
        y=feature_maps_embedded[:, 1],
        palette=sns.color_palette("hls", 1),
        legend="full",
        alpha=0.3
    )
    plt.savefig(os.path.join(args.model_folder,
                             't_sne_epoch{}.eps'.format(epoch)))


class DINOLoss(torch.nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, train=False):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        # raise NotImplementedError("TODO: load weight initialization")

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * torch.nn.functional.log_softmax(
                    student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        if train:
            self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / (len(teacher_output))
        # raise NotImplementedError("TODO: load weight initialization")

        # ema update
        self.center = self.center * self.center_momentum + \
            batch_center * (1 - self.center_momentum)


class MultiCropWrapper(torch.nn.Module):
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = torch.nn.Identity(), torch.nn.Identity()
        self.backbone = backbone
        self.head = head  # backbone.head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
