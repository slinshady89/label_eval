import numpy as np
import cv2
from drawing import print2img

from diagnostics import iou_rgb, acc_rgb, preprocess_inference, recall_rgb, precision_rgb, f1score_rgb, \
    colour_quota_rgb


## Evaluator for inferences of KITTI data structure
# sequences/XX/
# ---|image_2/xxxxxx.png
# ---|inference/xxxxxx.png
# ---|gt_labels/xxxxxx.png
class Evaluator(object):
    def __init__(self, _base_dir, _img_dir, _inf_label_dir, _gt_label_dir, _eval_list):
        self.base_dir_ = _base_dir
        self.img_dir_ = _img_dir
        self.inf_label_dir_ = _inf_label_dir
        self.gt_label_dir_ = _gt_label_dir
        self.eval_list = _eval_list
        self.threshold_ = np.array((69, 75, 110), dtype = np.int)

    def process_image(self, i):
        inf_label = cv2.imread(self.base_dir_ + self.inf_label_dir_ + self.eval_list[i])
        gt_label = cv2.imread(self.base_dir_ + self.gt_label_dir_ + self.eval_list[i])
        inf_label = cv2.resize(inf_label, (1024, 256))

        h, w, _ = gt_label.shape
        if h > 256 and w > 1024:
            gt_label = gt_label[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
        else:
            gt_label = gt_label

        inf_label_proc = preprocess_inference(inf = inf_label, threshold = self.threshold_)

        quota_gt = colour_quota_rgb(gt_label)
        recall = recall_rgb(gt_label, inf_label_proc)
        precision = precision_rgb(gt_label, inf_label_proc)

        prec_rec_qut = [[precision[0], recall[0], quota_gt[0]],
                        [precision[1], recall[1], quota_gt[1]],
                        [precision[2], recall[2], quota_gt[2]]]
        return prec_rec_qut

    def process_batch(self, q, begin, batch_size):
        prq = np.zeros((batch_size, 3, 3), dtype = np.float)
        for i in range(begin, begin + batch_size):
            print('processing image %d' % i)
            prq[i - begin] = self.process_image(i)
        q.put(prq)

    # def process_images(self, th, q):
    #     # print('evaluating %d images on threshold %d with pid %d' % (num_img - lower_bound, th, getpid()))
    #     PrecisionOverRecall = np.zeros((self.num_img_, 3, 3), dtype = np.float)
    #     # for k in range(self.lower_bound_, self.num_img_):
    #     #     PrecisionOverRecall[k] = self.process_image(i = k, q)
    #     por = np.zeros((3, 2), dtype = np.float)
    #     # calculates the mean of Precisions and Recall's over all images for each color channel and returns them without
    #     # colour quota
    #     por[0, :] = np.mean(PrecisionOverRecall[self.lower_bound_:self.num_img_, 0], axis = 0)[0:2]
    #     por[2, :] = np.mean(PrecisionOverRecall[self.lower_bound_:self.num_img_, 2], axis = 0)[0:2]
    #     # for the green channel the precision and recall is only estimated for images where the green quota is
    #     # larger than 7% (value estimated by a scatter plot)
    #     por[1, :] = np.mean(PrecisionOverRecall[PrecisionOverRecall[:, 1, 2] > 0.07, 1], axis = 0)[0:2]
    #     # print('threshold %d done by pid: %d' % (th, getpid()))
    #     q.put(por)

    def show_images(self, rgb_img, gt_label, inf_label, iou, recall, precision, quota_gt, quota_inf):
        vis_gt = cv2.addWeighted(gt_label / 255.0, 0.5, rgb_img / 255., 0.5, .0)
        vis_label = cv2.addWeighted(inf_label / 255.0, 0.5, rgb_img / 255., 0.5, .0)

        vis = cv2.vconcat([vis_gt, vis_label])
        text = 'IoU r: %.2f, g: %.2f, b: %.2f' % (iou[0], iou[1], iou[2])
        text_f1 = 'F1 r: %.2f, g: %.2f, b: %.2f' % f1score_rgb(gt_label, inf_label)
        text_acc = 'ACC r: %.2f, g: %.2f, b: %.2f' % acc_rgb(gt_label, inf_label)
        text_recall = 'Recall r: %.2f, g: %.2f, b: %.2f' % (recall[2], recall[1], recall[0])
        text_precision = 'Precision r: %.2f, g: %.2f, b: %.2f' % (precision[2], precision[1], precision[0])
        text_colour_gt = 'Quota r: %.2f, g: %.2f, b: %.2f' % (quota_gt[2], quota_gt[1], quota_gt[0])
        text_colour_inf = 'Quota r: %.2f, g: %.2f, b: %.2f' % (quota_inf[2], quota_inf[1], quota_inf[0])

        vis = print2img(vis, text, position = (10, 500))
        vis = print2img(vis, text_f1, position = (10, 460))
        vis = print2img(vis, text_acc, position = (10, 420))
        vis = print2img(vis, text_recall, position = (10, 380))
        vis = print2img(vis, text_precision, position = (10, 340))
        vis = print2img(vis, text_colour_gt, position = (10, 20))
        vis = print2img(vis, text_colour_inf, position = (10, 300))

        winname = 'concat'
        cv2.imshow(winname, vis)
        cv2.moveWindow(winname, 40, 20)

        cv2.waitKey(100)
