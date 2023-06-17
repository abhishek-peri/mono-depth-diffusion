import numpy as np
import cv2
import albumentations
import os
from tqdm import tqdm
from PIL import Image

def compute_errors(gt, pred):

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    log10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, log10

def preprocess_image(image_path):

    image = cv2.imread(image_path)
    preprocessor = preprocessor_fn(256)
    image = preprocessor(image=image)["image"]
    # image = (image/127.5 - 1.0).astype(np.float32)
    return image

def preprocessor_fn(size):
    rescaler = albumentations.SmallestMaxSize(max_size = size)
    cropper = albumentations.CenterCrop(height=size,width=size)
    preprocessor = albumentations.Compose([rescaler, cropper])
    return preprocessor

def main(gt_dir,pred_dir,gt_list,pred_list):

    abs_rel_total = 0.
    sq_rel_total = 0.
    rmse_total = 0.
    rmse_log_total = 0.
    log10_total = 0.
    a1_total = []
    a2_total = []
    a3_total = []

    for (gt_path,pred_path) in tqdm(zip(gt_list,pred_list)):
        # print(gt_dir+gt_path)
        # gt = cv2.imread(gt_dir+'/'+gt_path)
        gt = preprocess_image(gt_dir+'/'+gt_path)
        pred = cv2.imread(pred_dir+'/'+pred_path)

        # gt = gt[:,:,0]
        # pred = pred[:,:,0]

        gt = gt.astype('float32')
        gt /= (2**8 - 1)
        # gt *= 10.0

        pred = pred.astype('float32')
        pred /= (2**8 - 1)
        # pred *= 10.0

        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, log10 = compute_errors(gt,pred)
        abs_rel_total = abs_rel_total + abs_rel
        sq_rel_total = sq_rel_total + sq_rel
        rmse_total = rmse_total+rmse
        rmse_log_total = rmse_log_total+rmse_log
        log10_total = log10_total+log10
        a1_total.append(a1)
        a2_total.append(a2)
        a3_total.append(a3)

    print(f"abs_rel = {abs_rel_total/len(a3_total)}")
    print(f"sq_rel = {sq_rel_total/len(a3_total)}")
    print(f"rmse = {rmse_total/len(a3_total)}")
    print(f"rmse_log = {rmse_log_total/len(a3_total)}")
    print(f"log10 = {log10_total/len(a3_total)}")
    print(f"a1 {np.mean(a1_total)}")
    print(f"a2 {np.mean(a2_total)}")
    print(f"a3 {np.mean(a3_total)}")

if __name__=="__main__":
    gt_dir = "/root/data/test/depth" 
    gt_list = os.listdir(gt_dir)

    gt_list = gt_list[1:]

    pred_dir = "/root/data/logs/2023-06-02T21-32-24_dino-ldm-vq-4-semantic/checkpoints/samples/00877536/2023-06-14-00-43-41/img/sample1"
    pred_list = os.listdir(pred_dir)
    main(gt_dir,pred_dir,gt_list,pred_list)


