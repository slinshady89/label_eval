import numpy as np
import cv2

from diagnostics import iou_rgb, acc_rgb, preprocess_inference, recall_rgb, precision_rgb, f1score_rgb, \
    colour_quota_rgb, calc_best_th

def process_image(base_dir, label_dir, inf_dir, img_dir, i, threshhold_):
    inf_label = cv2.imread(base_dir + inf_dir + '%06d.png' % i)
    gt_label = cv2.imread(base_dir + label_dir + '%06d.png' % i)
    # gt_label = cv2.imread(base_dir + label_dir + '%06d.png' % i)
    rgb_img = cv2.imread(base_dir + img_dir + '%06d.png' % i)

    inf_label = cv2.resize(inf_label, (1024, 256))

    h, w, _ = gt_label.shape
    if h > 256 and w > 1024:
        gt_label = gt_label[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
        rgb_img = rgb_img[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
    else:
        gt_label = gt_label

    inf_label_proc = preprocess_inference(inf = inf_label, threshold = threshhold_)

    # iou = iou_rgb(gt_label, inf_label_proc)

    inf_bgr = inf_label  # preprocess_inference(inf_label, threshhold = 127)

    qb_gt, qg_gt, qr_gt = colour_quota_rgb(gt_label)
    # qr_inf, qg_inf, qr_inf = colour_quota_rgb(gt_label)
    b_recall, g_recall, r_recall = recall_rgb(gt_label, inf_label_proc)
    b_precision, g_precision, r_precision = precision_rgb(gt_label, inf_label_proc)

    # vis_gt = cv2.addWeighted(gt_label / 255.0, 0.5, rgb_img / 255., 0.5, .0)
    # vis_label = cv2.addWeighted(inf_label / 255.0, 0.5, rgb_img / 255., 0.5, .0)
    #
    # vis = cv2.vconcat([vis_gt, vis_label])
    # text = 'IoU r: %.2f, g: %.2f, b: %.2f' % (iou[0], iou[1], iou[2])
    # text_f1 = 'F1 r: %.2f, g: %.2f, b: %.2f' % f1score_rgb(gt_label, inf_bgr)
    # text_acc = 'ACC r: %.2f, g: %.2f, b: %.2f' % acc_rgb(gt_label, inf_bgr)
    # text_recall = 'Recall r: %.2f, g: %.2f, b: %.2f' % (b_recall, g_recall, r_recall)
    # text_precision = 'Precision r: %.2f, g: %.2f, b: %.2f' % (b_precision, g_precision, r_precision)
    # text_colour_gt = 'Quota r: %.2f, g: %.2f, b: %.2f' % (qr_gt, qg_gt, qr_gt)
    # text_colour_inf = 'Quota r: %.2f, g: %.2f, b: %.2f' % (qr_inf, qg_inf, qr_inf)
    #
    # vis = print2img(vis, text, position = (10, 500))
    # vis = print2img(vis, text_f1, position = (10, 460))
    # vis = print2img(vis, text_acc, position = (10, 420))
    # vis = print2img(vis, text_recall, position = (10, 380))
    # vis = print2img(vis, text_precision, position = (10, 340))
    # vis = print2img(vis, text_colour_gt, position = (10, 20))
    # vis = print2img(vis, text_colour_inf, position = (10, 300))
    #
    # winname = 'concat'
    # cv2.imshow(winname, vis)
    # cv2.moveWindow(winname, 40, 20)
    #
    # cv2.waitKey(100)
    return [[b_precision, b_recall, qb_gt], [g_precision, g_recall, qg_gt], [r_precision, r_recall, qr_gt]]
