import os
import random
import argparse
import torch
from pprint import pprint
from torchvision.transforms import *
from utils import check_dir
from models.pretraining_backbone import ResNet18Backbone
from data.pretraining import DataReaderPlainImg
from torch.utils.tensorboard import SummaryWriter


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-init', type=str,
                        default="")
    parser.add_argument("--size", type=int, default=256,
                        help="size of the images to feed the network")
    parser.add_argument('--output-root', type=str, default='results')
    args = parser.parse_args()

    args.output_folder = check_dir(
        os.path.join(args.output_root, "nearest_neighbors",
                     args.weights_init.replace("/", "_").replace("models", "")))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # model
    model = ResNet18Backbone(pretrained=False).cuda()
    model.load_state_dict(torch.load("/home/youssef/Uni/Deep_Learning_Lab/lr0.0005_bs64__local/models/ckpt_epoch9.pth",
                                     map_location=torch.device('cuda')), strict=False)    # raise NotImplementedError("TODO: build model and load weights snapshot")

    # dataset
    data_root = '/home/youssef/Uni/Deep_Learning_Lab/cv_assignment/data'
    val_transform = Compose(
        [Resize(args.size), CenterCrop((args.size, args.size)), ToTensor()])
    val_data = DataReaderPlainImg(os.path.join(
        data_root, str(args.size), "val"), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False,
                                             num_workers=2, pin_memory=True, drop_last=False)

    # raise NotImplementedError(
    #     "Load the validation dataset (crops), use the transform above.")

    # choose/sample which images you want to compute the NNs of.
    # You can try different ones and pick the most interesting ones.
    log = SummaryWriter(
        "/home/youssef/Uni/Deep_Learning_Lab/cv_assignment/tensorboard")
    query_indices = [101, 99, 23, 12]
    nns = []
    for idx, img in enumerate(val_loader):
        if idx not in query_indices:
            continue
        print("Computing NNs for sample {}".format(idx))
        closest_idx, closest_dist = find_nn(model, img, val_loader, 5)
        nns.append((val_loader.dataset[idx], [
                   val_loader.dataset[i] for i in closest_idx]))

    for idx, nn in enumerate(nns):
        images = torch.stack([nn[0]] + nn[1], dim=0)
        log.add_images("nearest_neighbor images", images, idx)
        log.flush()
    print("Done.")


def find_nn(model, query_img, loader, k):
    """
    Find the k nearest neighbors (NNs) of a query image, in the feature space of the specified mode.
    Args:
         model: the model for computing the features
         query_img: the image of which to find the NNs
         loader: the loader for the dataset in which to look for the NNs
         k: the number of NNs to retrieve
    Returns:
        closest_idx: the indices of the NNs in the dataset, for retrieving the images
        closest_dist: the L2 distance of each NN to the features of the query image
    """
    # raise NotImplementedError("TODO: nearest neighbors retrieval")
    model.eval()
    query_features = model.features(query_img.cuda())["out"]
    list_distances = []
    for idx, img in enumerate(loader):
        with torch.no_grad():
            features = model.features(img.cuda())["out"]
        distance = torch.dist(features, query_features).cpu()
        if distance != float(0):
            list_distances.append(distance)
    closest_dist, closest_idx = torch.stack(
        list_distances, dim=0).topk(k=k, largest=False, sorted=True)

    return closest_idx, closest_dist


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
