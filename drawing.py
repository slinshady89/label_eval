import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot, colorbar, colors, cm


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