from PIL import Image
from skimage.measure import compare_ssim
import math
import numpy as np
import openface
import os
import argparse
import cv2

dataroot = '/dataset/AffineGAN_dataset/'
conv_pattern = 'img_{0:04d}.png'
videoGAN_pattern = 'gen_1{0:04d}.png'
flow_pattern = '001_{0:02d}_pred.png'
ganim_pattern = '{0:d}_out.jpg'
conv_start = 1
conv_end = 10
videoGAN_start = 1
videoGAN_end = 17
flow_start = 0
flow_end = 17
ours_start = 0
ours_end = 16
ganim_start = 0
ganim_end = 16

size = 128
gray = 'RGB'
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '.', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
parser = argparse.ArgumentParser()

parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--mode', type=str, default='train')

args = parser.parse_args()
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
align = openface.AlignDlib(args.dlibFacePredictor)


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def get_ssim_psnr(ref_dir, pred_dir, pattern, start, end):
    ssim_conv = 0.
    psnr_conv = 0.
    imgs = sorted(os.listdir(ref_dir))
    img_num = len(imgs)
    for i in range(start, end):
        a = Image.open(
            os.path.join(ref_dir, imgs[int((i - 1) * img_num / (end - start))]).convert(gray).resize((size, size)))
        a = np.array(a)
        b = Image.open(pred_dir + pattern.format(i)).convert(gray).resize((size, size))
        b = np.array(b)
        ssim_conv += compare_ssim(a, b, multichannel=gray == 'RGB')
        psnr_conv += psnr(b, a)
    return ssim_conv / (end - start), psnr_conv / (end - start)


def get_rep(img_path):
    bgrImg = cv2.imread(img_path)
    bgrImg = cv2.resize(bgrImg, (size, size))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        return np.random.rand(128)
    alignedFace = align.align(args.imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    rep1 = net.forward(alignedFace)
    return rep1


def get_ACD(ref_dir, pred_dir, pattern, start, end):
    ACD_m = 0.
    ACD_i = 0.
    ACD_c = 0.
    ACD_ref_i = 0.
    ACD_ref_c = 0.
    imgs = sorted(os.listdir(ref_dir))
    img_num = len(imgs)
    first = get_rep(os.path.join(ref_dir, imgs[0]))
    pred_features = []
    ref_features = []
    for i in range(start, end):
        a = get_rep(os.path.join(ref_dir, imgs[int((i - 1) * img_num / (start - end))]))
        b = get_rep(os.path.join(pred_dir, pattern.format(i)))
        pred_features.append(b)
        ref_features.append(a)
        ACD_m += np.linalg.norm(a - b, 2)
        ACD_i += np.linalg.norm(b - first, 2)
        ACD_ref_i += np.linalg.norm(a - first, 2)
    for i in range(len(pred_features)):
        for j in range(i, len(pred_features)):
            ACD_c += np.linalg.norm(pred_features[i] - pred_features[j], 2)
            ACD_ref_c += np.linalg.norm(ref_features[i] - ref_features[j], 2)
    return ACD_m / (end - start), ACD_i / (end - start), ACD_c / (
                len(pred_features) * (len(pred_features) - 1)) * 2, ACD_ref_i / (end - start), ACD_ref_c / (
                       len(pred_features) * (len(pred_features) - 1)) * 2

results_root = '/dataset/Results'
acd_m = [0.] * 5
acd_i = [0.] * 5
acd_c = [0.] * 5
acd_ref_i = [0.] *5
acd_ref_c = [0.] * 5
img_num = 0
for category in os.listdir(dataroot):
    print(category, 'begin')
    ref_root = os.path.join(dataroot, category, 'test', 'img')
    imgs = sorted([f for f in os.listdir(ref_root)])
    img_num += len(imgs)
    for idx in range(len(imgs)):
        id = imgs[idx]
        ours_pattern = id + '_fake_B_list{0:d}.png'
        ref_dir = os.path.join(ref_root, id)
        conv_pred_dir = os.path.join(results_root, 'convlstm', 'results_valid', category, 'vid_{0:04d}'.format(idx + 1))
        videoGAN_pred_dir = os.path.join(results_root, 'videogan', 'results_valid', category, 'vis')
        flow_pred_dir = os.path.join(results_root, 'flowground', 'results_valid', category, id)
        ours_pred_dir = os.path.join(results_root, 'ours', 'results_valid', category, 'test_latest', 'images')
        ganim_pred_dir = os.path.join(results_root, 'ganimation', 'results_valid', category, id)
        acds_conv = get_ACD(ref_dir, conv_pred_dir, conv_pattern, conv_start, conv_end)
        acds_video = get_ACD(ref_dir, videoGAN_pred_dir, videoGAN_pattern, videoGAN_start, videoGAN_end)
        acds_flow = get_ACD(ref_dir, flow_pred_dir, flow_pattern, flow_start, flow_end)
        acds_ganim = get_ACD(ref_dir, ganim_pred_dir, ganim_pattern, ganim_start, ganim_end)
        acds_ours = get_ACD(ref_dir, ours_pred_dir, ours_pattern, ours_start, ours_end)
        acd_m[0] += acds_conv[0]
        acd_m[1] += acds_video[0]
        acd_m[2] += acds_flow[0]
        acd_m[3] += acds_ganim[0]
        acd_m[4] += acds_ours[0]
        acd_i[0] += acds_conv[1]
        acd_i[1] += acds_video[1]
        acd_i[2] += acds_flow[1]
        acd_i[3] += acds_ganim[1]
        acd_i[4] += acds_ours[1]
        acd_c[0] += acds_conv[2]
        acd_c[1] += acds_video[2]
        acd_c[2] += acds_flow[2]
        acd_c[3] += acds_ganim[2]
        acd_c[4] += acds_ours[2]
        acd_ref_i[0] += acds_conv[3]
        acd_ref_i[1] += acds_video[3]
        acd_ref_i[2] += acds_flow[3]
        acd_ref_i[3] += acds_ganim[3]
        acd_ref_i[4] += acds_ours[3]
        acd_ref_c[0] += acds_conv[4]
        acd_ref_c[1] += acds_video[4]
        acd_ref_c[2] += acds_flow[4]
        acd_ref_c[3] += acds_ganim[4]
        acd_ref_c[4] += acds_ours[4]
    print(category, 'done')

print(np.array(acd_m) / img_num)
print(np.array(acd_i) / img_num)
print(np.array(acd_c) / img_num)
print(np.array(acd_ref_i) / img_num)
print(np.array(acd_ref_c) / img_num)
