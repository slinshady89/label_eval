import cv2
import matplotlib
from diagnostics import calc_iou_rgb, calc_acc_rgb, preprocess_inference, recall_rgb, precision_rgb, f1score_rgb




def print2img(img, text, position):

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = position
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

    h, w, _ = gt_label.shape
    if h > 256 and w > 1024:
        gt_label = gt_label[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
        rgb_img = rgb_img[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
    else:
        gt_label = gt_label

    inf_label_proc = preprocess_inference(inf_label, threshhold = 127)

    vis_gt = cv2.addWeighted(gt_label / 255.0, 0.5, rgb_img / 255., 0.5, .0)
    vis_label = cv2.addWeighted(inf_label / 255.0, 0.5, rgb_img / 255., 0.5, .0)

    vis = cv2.vconcat([vis_gt, vis_label])

    iou = calc_iou_rgb(gt_label, inf_label_proc)

    inf_bgr = inf_label #preprocess_inference(inf_label, threshhold = 127)

    h, w, _ = gt_label.shape

    sum = 0
    num = 0
    for u in range(0, w-1):
        for v in range(0, h-1):
            if gt_label[v, u, 1] > 0:
                sum += (gt_label[v, u, 1] - inf_bgr[v, u, 1]) / gt_label[v, u, 1]
                num += 1

    amount = sum / num

    text = 'IoU r: %.2f, b: %.2f, g: %.2f      |      g_amount: %.2f' % (iou[0], iou[2], iou[1], amount)
    text_acc = 'Acc r: %.2f, b: %.2f, g: %.2f' % calc_acc_rgb(gt_label, inf_bgr)

    vis = print2img(vis, text, position = (10, 500))
    vis = print2img(vis, text_acc, position = (10, 460))

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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Finished")