import cv2
from diagnostics import iou_rgb, acc_rgb, preprocess_inference, recall_rgb, precision_rgb, f1score_rgb, colour_quota_rgb
from matplotlib import pyplot, colorbar, colors, cm
import numpy as np
import matplotlib


def print2img(img, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = position
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 1

    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    return img


def draw_por_colorbar(array3d):
    # cmap ref: https://matplotlib.org/examples/color/colormaps_reference.html
    cmap = cm.get_cmap("plasma")

    normalize = matplotlib.colors.Normalize(vmin = min(array3d[:, 2]), vmax = max(array3d[:, 2]))
    colors = [cmap(normalize(value)) for value in array3d[:, 2]]

    fig, ax = pyplot.subplots()
    ax.scatter(array3d[:, 0], array3d[:, 1], color = colors, marker = '.', s = 1.5)
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.ylim(0, 1)
    pyplot.xlim(0, 1)
    cax, _ = colorbar.make_axes(ax)
    colorbar.ColorbarBase(cax, cmap = cmap, norm = normalize)
    pyplot.xlabel('Quota \ngreen')
    pyplot.ylabel('Quota')


def draw_bin_coloured(array3d):
    num, channels = array3d.shape
    if channels == 3:
        b_por_6 = array3d[:, 2] < 0.07
        por_6 = array3d[b_por_6, :]
        por_7 = array3d[np.invert(b_por_6), :]
        pyplot.plot(por_6[:, 1], por_6[:, 0], linestyle = 'None', marker = '.', markersize = 1.5, color = 'r')
        pyplot.plot(por_7[:, 1], por_7[:, 0], linestyle = 'None', marker = '.', markersize = 1.5, color = 'g')
        pyplot.ylim(0, 1)
        pyplot.xlim(0, 1)
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
    else: # only 2D array without colour information
        pyplot.plot(array3d[:, 1], array3d[:, 0], linestyle = 'None', marker = '.', markersize = 1.5, color = 'g')
        pyplot.ylim(0, 1)
        pyplot.xlim(0, 1)
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')


def process_image(base_dir, inf_dir, label_dir, img_dir, i, threshhold_ = 120):
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

    inf_label_proc = preprocess_inference(inf = inf_label, threshhold = threshhold_)

    vis_gt = cv2.addWeighted(gt_label / 255.0, 0.5, rgb_img / 255., 0.5, .0)
    vis_label = cv2.addWeighted(inf_label / 255.0, 0.5, rgb_img / 255., 0.5, .0)

    vis = cv2.vconcat([vis_gt, vis_label])

    iou = iou_rgb(gt_label, inf_label_proc)

    inf_bgr = inf_label  # preprocess_inference(inf_label, threshhold = 127)

    qb_gt, qg_gt, qr_gt = colour_quota_rgb(gt_label)
    qr_inf, qg_inf, qr_inf = colour_quota_rgb(gt_label)
    b_recall, g_recall, r_recall = recall_rgb(gt_label, inf_bgr)
    b_precision, g_precision, r_precision = precision_rgb(gt_label, inf_bgr)

    text = 'IoU r: %.2f, g: %.2f, b: %.2f' % (iou[0], iou[1], iou[2])
    text_f1 = 'F1 r: %.2f, g: %.2f, b: %.2f' % f1score_rgb(gt_label, inf_bgr)
    text_acc = 'ACC r: %.2f, g: %.2f, b: %.2f' % acc_rgb(gt_label, inf_bgr)
    text_recall = 'Recall r: %.2f, g: %.2f, b: %.2f' % (b_recall, g_recall, r_recall)
    text_precision = 'Precision r: %.2f, g: %.2f, b: %.2f' % (b_precision, g_precision, r_precision)
    text_colour_gt = 'Quota r: %.2f, g: %.2f, b: %.2f' % (qr_gt, qg_gt, qr_gt)
    text_colour_inf = 'Quota r: %.2f, g: %.2f, b: %.2f' % (qr_inf, qg_inf, qr_inf)

    #
    # vis = print2img(vis, text, position = (10, 500))
    # vis = print2img(vis, text_f1, position = (10, 460))
    # vis = print2img(vis, text_acc, position = (10, 420))
    # vis = print2img(vis, text_recall, position = (10, 380))
    # vis = print2img(vis, text_precision, position = (10, 340))
    # vis = print2img(vis, text_colour_gt, position = (10, 20))
    # vis = print2img(vis, text_colour_inf, position = (10, 300))
    #
    #
    # winname = 'concat'
    # cv2.imshow(winname, vis)
    # cv2.moveWindow(winname, 40, 20)
    #
    # cv2.waitKey(100)
    return [[b_precision, b_recall, qb_gt], [g_precision, g_recall, qg_gt], [r_precision, r_recall, qr_gt]]


def main():
    base_dir = '/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/08/'
    # inf_dir = '20190808_160355/'
    # inf_dir = '20190730_170755/'
    inf_dir = 'inference/'
    label_dir = 'labels_gt/'
    img_dir = 'image_2/'

    num_img = 4000
    precision_over_th = np.zeros((255, 1), dtype = np.float)
    recall_over_th = np.zeros((255, 1), dtype = np.float)

    lower_bound = 0
    for th in range(5, 256, 50):
        PrecisionOverRecall = process_images(base_dir, img_dir, inf_dir, label_dir, lower_bound,
                                             num_img, th)
        print('threshold %d done' % th)
        print(PrecisionOverRecall.shape)
        precision_over_th[th] = np.mean(PrecisionOverRecall[PrecisionOverRecall[:, 1, 2] > 0.07, 0])
        recall_over_th[th] = np.mean(PrecisionOverRecall[PrecisionOverRecall[:, 1, 2] > 0.07, 1])

    por = np.hstack((precision_over_th, recall_over_th))

    with open('/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/Data/por_large.txt', "w") as f:
        np.savetxt(f, por, fmt = '%2.6f')

    f.close()

    with open('/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/Data/por_large.txt', "r") as f:
        test_read = np.loadtxt(f)

    draw_bin_coloured(test_read)
    pyplot.plot(por[:, 1], por[:, 0], linestyle = 'None',
                marker = '.', markersize = 1.5, color = 'r')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.title('Precision over Recall means for thresholds of binary cutting')

    pyplot.show()


def process_images(base_dir, img_dir, inf_dir, label_dir, lower_bound, num_img, th):
    PrecisionOverRecall = np.zeros((num_img, 3, 3), dtype = np.float)
    for k in range(lower_bound, num_img):
        PrecisionOverRecall[k] = process_image(base_dir = base_dir, label_dir = label_dir, inf_dir = inf_dir,
                                            img_dir = img_dir,
                                            i = k, threshhold_ = th)
    return PrecisionOverRecall


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Finished")
