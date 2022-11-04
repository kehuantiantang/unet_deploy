import os
import os.path as osp
import cv2

def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)  # connectivity参数的默认值为8
    stats = stats[stats[:, 4].argsort()]
    return stats[:-1]  # 排除最外层的连通图*

def make_bbox_label(y_pred_path,save_txt_path):

    names = []
    y_pred_path = y_pred_path
    save_txt_path = save_txt_path


    for name in os.listdir(y_pred_path):
        if name.split('.')[-1].lower() in ['tif', 'jpg', 'png']:
            names.append(name.split('.')[0])

    for name in names:
        # mask_img = osp.join(y_pred_path, '%s.jpg' % name)
        mask_img = osp.join(y_pred_path, '%s.png' % name)
        save_txt = osp.join(save_txt_path, '%s.txt' % name)
        if osp.exists(mask_img):
            mask = cv2.imread(mask_img, cv2.COLOR_BGR2GRAY)

    # # 获取mask（灰度图）
    # mask = cv2.imread(r'D:\jupyter\1019\f1\gt\base_zl_20_tx_897896_ty_410694__from_4S_201_221_06.png', cv2.COLOR_BGR2GRAY)
    # 转换成二值图
            ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

            bboxs = mask_find_bboxs(mask)
            # print(bboxs)

            for b in bboxs:
                if b[2]>10 and b[3]>10:
                    x0, y0 = b[0], b[1]
                    x1 = b[0] + b[2]
                    y1 = b[1] + b[3]
                    # print(f'x0:{x0}, y0:{y0}, x1:{x1}, y1:{y1}')
                    start_point, end_point = (x0, y0), (x1, y1)
                    xcenter = (x1 - x0)/2 + x0
                    ycenter = (y1 - y0)/2 + y0
                    w = b[2]
                    h = b[3]

                    fsave = open(save_txt, mode='a')
                    fsave.write(f'0 {xcenter} {ycenter} {w} {h}' + "\n")
                    fsave.close()
                else:
                    fsave = open(save_txt, mode='a')
                    fsave.close()

        # color = (0, 0, 255)  # Red color in BGR；红色：rgb(255,0,0)
        # thickness = 1  # Line thickness of 1 px
        # mask_BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 转换为3通道图，使得color能够显示红色。
        # mask_bboxs = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)

        # """
        # # Displaying the image
        # cv2.imshow('show_image', mask_bboxs)
        # cv2.waitKey(0)
        # """
        # cv2.imwrite(r'D:\jupyter\1019\f1\gb\base_zl_20_tx_897896_ty_410694__from_4S_201_221_06.png', mask_bboxs)

# class, xcenter, ycenter, w, h

def make_bbox_mask(y_pred_path, save_txt_path):
    names = []
    y_pred_path = y_pred_path
    save_txt_path = save_txt_path

    # y_pred_path = 'E:/pwd/unet/'
    # y_pred_path = 'E:/pwd/tp_0929/mask/'
    # save_txt_path = 'D:/jupyter/1019/f1/unet/'
    # save_txt_path = 'D:/jupyter/1019/f1/label/'

    for name in os.listdir(y_pred_path):
        if name.split('.')[-1].lower() in ['tif', 'jpg', 'png']:
            names.append(name.split('.')[0])

    for name in names:
        mask_img = osp.join(y_pred_path, '%s.jpg' % name)
        # mask_img = osp.join(y_pred_path, '%s.png' % name)
        save_txt = osp.join(save_txt_path, '%s.txt' % name)
        if osp.exists(mask_img):
            mask = cv2.imread(mask_img, cv2.COLOR_BGR2GRAY)

            # # 获取mask（灰度图）
            # mask = cv2.imread(r'D:\jupyter\1019\f1\gt\base_zl_20_tx_897896_ty_410694__from_4S_201_221_06.png', cv2.COLOR_BGR2GRAY)
            # 转换成二值图
            ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

            bboxs = mask_find_bboxs(mask)
            # print(bboxs)

            for b in bboxs:
                if b[2] > 10 and b[3] > 10:
                    x0, y0 = b[0], b[1]
                    x1 = b[0] + b[2]
                    y1 = b[1] + b[3]
                    # print(f'x0:{x0}, y0:{y0}, x1:{x1}, y1:{y1}')
                    start_point, end_point = (x0, y0), (x1, y1)
                    xcenter = (x1 - x0) / 2 + x0
                    ycenter = (y1 - y0) / 2 + y0
                    w = b[2]
                    h = b[3]

                    fsave = open(save_txt, mode='a')
                    fsave.write(f'0 {xcenter} {ycenter} {w} {h}' + "\n")
                    fsave.close()
                else:
                    fsave = open(save_txt, mode='a')
                    fsave.close()