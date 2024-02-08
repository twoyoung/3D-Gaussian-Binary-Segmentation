import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as tf
import json
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import os

def binarize_mask(mask, threshold=0.9):
    """Binarize a grayscale mask based on a threshold"""
    binary_mask = (mask >= threshold).type(torch.float32)
    return binary_mask

def calculate_biou(pred_mask, gt_mask):
    """Calculate Binary Intersection over Union (BIoU) for a single image"""
    intersection = torch.logical_and(pred_mask, gt_mask)
    union = torch.logical_or(pred_mask, gt_mask)
    biou_score = torch.sum(intersection) / torch.sum(union)
    return biou_score

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                biou_scores = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    # ... [existing metric calculations]

                    # Binarize and calculate BIoU
                    binarized_render = binarize_mask(renders[idx])
                    biou_score = calculate_biou(binarized_render, gts[idx])
                    biou_scores.append(biou_score.item())

                # Output BIoU along with other metrics
                print("  BIoU : {:>12.7f}".format(np.mean(biou_scores)))
                print("")

                # Add BIoU to the dictionary
                full_dict[scene_dir][method].update({"BIoU": np.mean(biou_scores)})
                per_view_dict[scene_dir][method].update({"BIoU": {name: biou for biou, name in zip(biou_scores, image_names)}})

                            # ... [rest of your code]

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
