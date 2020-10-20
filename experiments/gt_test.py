import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

root = str(Path(__file__).resolve().parent.parent)
def main():


    path = os.path.join(root, 'label_generator', 'data')
    path2 = os.path.join(root, 'data_generation', 'data')
    gt_path = os.path.join(root, 'experiments', 'data', 'gt_test')
    classes = [d for d in list(os.listdir(gt_path)) if os.path.isdir(os.path.join(gt_path, d))]
    print('classes', classes)
    gt_path = os.path.join(root, 'experiments', 'data', 'gt_test')

    plot_labels = False
    c0 = 0.7
    c1 = 1.0 - c0
    ch1 = 2  # color pred
    ch2 = 0  # color new pred and gen


    ious_gt_vs_new_pred = []
    ious_gt_vs_pred = []
    ious_pred_vs_new_pred = []
    ious_gt_vs_gen = []
    ious_gen_vs_pred = []
    ious_1 = 0
    ious_2 = 0
    ious_3 = 0
    ious_4 = 0
    ious_5 = 0
    total_samples = 0

    rotations = ['foreground', 'foreground180']
    for i, cls in enumerate(classes):
        #if cls != 'Joint':
        #    continue

        for j, rot in enumerate(rotations):
            print('cls {}/{}, rot {}/{}'.format(i+1, len(classes), j+1, len(rotations)))
            rot_path = os.path.join(path, cls, rot)
            rot_path2 = os.path.join(path2, cls, rot)
            gt_rot_path = os.path.join(gt_path, cls, rot)

            tag = '.color.mask.0.png'
            samples = [s[:-len(tag)] for s in list(os.listdir(gt_rot_path)) if tag in s]

            for sample in samples:
                gt_sample_path = os.path.join(gt_rot_path, '{}{}'.format(sample, tag))
                new_pred_path = os.path.join(rot_path, '{}.new_pred.label.png'.format(sample))
                pred_path = os.path.join(rot_path, '{}.pred.label.png'.format(sample))
                gen_path = os.path.join(rot_path, '{}.gen.label.png'.format(sample))
                image_path = os.path.join(rot_path2, '{}.color.png'.format(sample))


                gt_label = np.array(Image.open(gt_sample_path).convert('RGB'), dtype=np.uint8)[:, :, 0]
                new_pred_label = np.array(Image.open(new_pred_path).convert('RGB'), dtype=np.uint8)[:, :, 0]
                pred_label = np.array(Image.open(pred_path).convert('RGB'), dtype=np.uint8)[:, :, 0]
                gen_label = np.array(Image.open(gen_path).convert('RGB'), dtype=np.uint8)[:, :, 0]
                image = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)

                if plot_labels:
                    plt.subplot(2, 2, 1)
                    plt.imshow(image)
                    plt.axis('off')
                    plt.title('RGB Image')
                    plt.subplot(2, 2, 2)
                    plt.imshow(gt_label)
                    plt.axis('off')
                    plt.title('Human Hand annotation')
                    plt.subplot(2, 2, 3)
                    plt.imshow(pred_label)
                    plt.axis('off')
                    plt.title('Background Subtraction')
                    plt.subplot(2, 2, 4)
                    plt.imshow(new_pred_label)
                    plt.axis('off')
                    plt.title('Segmentation Model')
                    plt.show()

                if plot_labels:
                    plt.subplot(2, 2, 1)
                    plt.imshow(image)
                    plt.axis('off')
                    plt.title('RGB Image')
                    plt.subplot(2, 2, 2)
                    plt.imshow(gt_label)
                    plt.axis('off')
                    plt.title('Human Hand annotation')
                    plt.subplot(2, 2, 3)
                    plt.imshow(gen_label)
                    plt.axis('off')
                    plt.title('Classical Approach')
                    plt.subplot(2, 2, 4)
                    plt.imshow(pred_label)
                    plt.axis('off')
                    plt.title('Deep Learning Approach')
                    plt.show()

                if plot_labels:
                    added = image.copy()

                    added[:, :, ch1][pred_label != 0] = added[:, :, ch1][pred_label != 0] * c0 + pred_label[pred_label != 0] * c1
                    added[:, :, ch2][new_pred_label != 0] = added[:, :, ch2][new_pred_label != 0] * c0 + new_pred_label[new_pred_label != 0] * c1
                    plt.imshow(added)
                    plt.axis('off')
                    plt.title('Background Subtraction vs Segmentation Model')
                    plt.show()

                    added = image.copy()
                    added[:, :, ch1][pred_label != 0] = added[:, :, ch1][pred_label != 0] * c0 + pred_label[pred_label != 0] * c1
                    added[:, :, ch2][gen_label != 0] = added[:, :, ch2][gen_label != 0] * c0 + gen_label[gen_label != 0] * c1
                    plt.imshow(added)
                    plt.axis('off')
                    plt.title('Deep Learning vs Classical')
                    plt.show()

                ious_gt_vs_new_pred.append(compute_IoU(gt_label, new_pred_label))
                ious_gt_vs_pred.append(compute_IoU(gt_label, pred_label))
                ious_pred_vs_new_pred.append(compute_IoU(pred_label, new_pred_label))
                ious_gt_vs_gen.append(compute_IoU(gt_label, gen_label))
                ious_gen_vs_pred.append(compute_IoU(gen_label, pred_label))

                if ious_gt_vs_new_pred[-1][0] >= 0.5:
                    ious_1 += 1
                if ious_gt_vs_pred[-1][0] >= 0.5:
                    ious_2 += 1
                if ious_pred_vs_new_pred[-1][0] >= 0.5:
                    ious_3 += 1
                if ious_gt_vs_gen[-1][0] >= 0.5:
                    ious_4 += 1
                if ious_gen_vs_pred[-1][0] >= 0.5:
                    ious_5 += 1
                total_samples += 1

            iou_gt_vs_new_pred = np.round(np.mean(np.array(ious_gt_vs_new_pred), axis=0), 4)
            iou_gt_vs_pred = np.round(np.mean(np.array(ious_gt_vs_pred), axis=0), 4)
            iou_pred_vs_new_pred = np.round(np.mean(np.array(ious_pred_vs_new_pred), axis=0), 4)
            iou_gt_vs_gen = np.round(np.mean(np.array(ious_gt_vs_gen), axis=0), 4)
            iou_gen_vs_pred = np.round(np.mean(np.array(ious_gen_vs_pred), axis=0), 4)

            ious_1_out = np.round(ious_1/total_samples, 4)
            ious_2_out = np.round(ious_2/total_samples, 4)
            ious_3_out = np.round(ious_3/total_samples, 4)
            ious_4_out = np.round(ious_4/total_samples, 4)
            ious_5_out = np.round(ious_5/total_samples, 4)

            print('gt_vs_new_pred:      iou = {}, accuracy = {}, precision = {}, recall = {}, iou >= 0.5: {}'.format(*iou_gt_vs_new_pred, ious_1_out))
            print('gt_vs_pred:          iou = {}, accuracy = {}, precision = {}, recall = {}, iou >= 0.5: {} '.format(*iou_gt_vs_pred, ious_2_out))
            print('pred_vs_new_pred:    iou = {}, accuracy = {}, precision = {}, recall = {}, iou >= 0.5: {} '.format(*iou_pred_vs_new_pred, ious_3_out))
            print('gt_vs_gen:           iou = {}, accuracy = {}, precision = {}, recall = {}, iou >= 0.5: {} '.format(*iou_gt_vs_gen, ious_4_out))
            print('gen_vs_pred:         iou = {}, accuracy = {}, precision = {}, recall = {}, iou >= 0.5: {} '.format(*iou_gen_vs_pred, ious_5_out))



def compute_IoU(ground_truth, label):
    # flatten image, set 255 to 1 and background to 3 such that we can count tp, fp, fn, tn
    ground_truth_flat = np.ndarray.flatten(ground_truth)
    ground_truth_flat[ground_truth_flat != 0] = 1
    ground_truth_flat[ground_truth_flat == 0] = 3

    # flatten label as well
    label_flat = np.ndarray.flatten(label)
    label_flat[label_flat != 0] = 1

    # compute difference, and get uniques and their counts
    diff = ground_truth_flat - label_flat
    unique, counts = np.unique(diff, return_counts=True)

    # count tp, fp and fn
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for j, i in enumerate(unique):
        if i == 0:
            tp = counts[j]
        elif i == 1:
            fp = counts[j]
        elif i == 2:
            fn = counts[j]
        elif i == 3:
            tn = counts[j]


    iou = float(float(tp) / float(tp + fp + fn))
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = float(float(tp) / float(tp + fp))
    recall = float(float(tp) / float(tp + fn))
    return [iou, accuracy, precision, recall]


def change_contrast(image):

    #print('change contrast')
    #cv2.imshow('input', np.array(image, dtype=np.uint8))
    lab = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_BGR2LAB)
    #cv2.imshow("lab", lab)

    l, a, b = cv2.split(lab)
    #cv2.imshow('l_channel', l)
    #cv2.imshow('a_channel', a)
    #cv2.imshow('b_channel', b)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    #cv2.imshow('CLAHE output', cl)
    limg = cv2.merge((cl, a, b))
    #cv2.imshow('limg', limg)

    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    #cv2.imshow('final', image)

    return np.array(image, dtype=np.uint8)



if __name__ == '__main__':
    main()