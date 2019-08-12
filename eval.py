import cv2
import numpy as np


def iou_for_semantic_class(target_color_channel, inferenced_color_channel):
    intersection = np.logical_and(target_color_channel, inferenced_color_channel)
    union = np.logical_or(target_color_channel, inferenced_color_channel)
    return np.sum(intersection) / np.sum(union)


def calc_iou_rgb(img_1, img_2):
    b = iou_for_semantic_class(img_1[:, :, 0], img_2[:, :, 0])
    g = iou_for_semantic_class(img_1[:, :, 1], img_2[:, :, 1])
    r = iou_for_semantic_class(img_1[:, :, 2], img_2[:, :, 2])
    return r, g, b


def process_inf(inf, threshhold):
    _, b = cv2.threshold(inf[:, :, 0], threshhold, 255, cv2.THRESH_BINARY)
    _, g = cv2.threshold(inf[:, :, 1], threshhold, 255, cv2.THRESH_BINARY)
    _, r = cv2.threshold(inf[:, :, 2], threshhold, 255, cv2.THRESH_BINARY)
    inf[:, :, 0] = b
    inf[:, :, 1] = g
    inf[:, :, 2] = r
    return inf


def print2img(img, text):

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    return img


def process_image(base_dir, inf_dir, label_dir, img_dir, i):
    inf_label = cv2.imread(base_dir + inf_dir + '%06d.png' % i)
    gt_label = cv2.imread(base_dir + label_dir + '%06d.png' % i)
    rgb_img = cv2.imread(base_dir + img_dir + '%06d.png' % i)

    inf_label = cv2.resize(inf_label, (1024, 256))

    h, w, c = gt_label.shape
    if h > 256 and w > 1024:
        gt_label = gt_label[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
        rgb_img = rgb_img[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
    else:
        gt_label = gt_label

    inf_label_proc = process_inf(inf_label, threshhold = 127)

    vis_gt = cv2.addWeighted(gt_label / 255.0, 0.5, rgb_img / 255., 0.5, .0)
    vis_label = cv2.addWeighted(inf_label / 255.0, 0.5, rgb_img / 255., 0.5, .0)

    vis = cv2.vconcat([vis_gt, vis_label])

    iou = calc_iou_rgb(gt_label, inf_label_proc)

    inf_bgr = process_inf(inf_label, threshhold = 50)

    h, w, _ = gt_label.shape

    sum = 0
    num = 0
    for u in range(0, w-1):
        for v in range(0, h-1):
            if gt_label[v, u, 1] > 0:
                sum += (gt_label[v, u, 1] - inf_bgr[v, u, 1]) / gt_label[v, u, 1]
                num += 1

    amount = sum / num

    text = 'IoU r: %.2f, b: %.2f, g: %.2f        |      g_amount: %.2f' % (iou[0], iou[2], iou[1], amount)

    vis = print2img(vis, text)

    winname = 'concat'
    cv2.imshow(winname, vis)
    cv2.moveWindow(winname, 40, 20)

    cv2.waitKey(50)
    return iou, amount


def main():
    base_dir = '/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/08/'
    inf_dir = '20190808_160355/'
    label_dir = 'labels_gt/'
    img_dir = 'image_2/'

    for k in range(0, 4000):
        iou, amount = process_image(base_dir = base_dir, label_dir = label_dir, inf_dir = inf_dir, img_dir = img_dir, i = k)
        print(iou[1])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Finished")