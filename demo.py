import argparse
import os
import platform
import sys
from pathlib import Path
import copy
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode



import os.path as osp
import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from loguru import logger

sys.path.append(str(ROOT / 'botsort'))


from yolox.utils.visualize import plot_tracking
from tracker.bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmpdir', type=str, help='path for tmp images') # 指定图片存放位置
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save', action='store_true', help='save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    
    #botsort
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16", dest="fp16", default=True, action="store_true",help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=True, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.5, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=100, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=10, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="fuse_score", default=True, action="store_true", help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="test mot20.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=os.path.join(ROOT, r"botsort/fast_reid/configs/MOT17/sbs_S50.yml"), type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=os.path.join(ROOT, r"botsort/pretrained/mot17_sbs_S50.pth"), type=str,help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')
    
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    
    opt.ablation = False
    opt.mot20 = not opt.fuse_score
    
    print_args(vars(opt))
    return opt


def del_file(path_data):
    for i in os.listdir(path_data):
        file_data = path_data + "/" + i
        if os.path.isfile(file_data) == True:
            os.remove(file_data)
        else:
            del_file(file_data)


@smart_inference_mode()

def run_by_video(model, opt):  
    tmpdir = str(opt.tmpdir) # 中间文件存放位置
    
    stride, names, pt = model.stride, model.names, model.pt
    opt.imgsz = check_img_size(opt.imgsz, s=stride)  # check image size
    opt.source = str(opt.source)
    save_img = opt.save and not opt.source.endswith('.txt')  # save inference images
    is_file = Path(opt.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = opt.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = opt.source.isnumeric() or opt.source.endswith('.txt') or (is_url and not is_file)
    screenshot = opt.source.lower().startswith('screen')
    if is_url and is_file:
        opt.source = check_file(opt.source)  # download

    # Directories
    opt.name = Path(opt.source).stem
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=True)  # increment run
    (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    del_file(str(save_dir))

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        opt.view_img = check_imshow()
        dataset = LoadStreams(opt.source, img_size=opt.imgsz, stride=stride, auto=pt, vid_stride=opt.vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(opt.source, img_size=opt.imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(opt.source, img_size=opt.imgsz, stride=stride, auto=pt, vid_stride=opt.vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    cap = cv2.VideoCapture(opt.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    save_folder = osp.join(save_dir, 'track')
    os.makedirs(save_folder, exist_ok=True)

    tracker = BoTSORT(opt)
    save_path0 = str(save_dir / 'track' / opt.name)  # im.jpg
    save_path0 = str(Path(save_path0).with_suffix('.mp4'))
    vid_writer0 = cv2.VideoWriter(save_path0, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_id = 0
    results = []
    results_list = []
    
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *opt.imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            opt.visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
            pred = model(im, augment=opt.augment, visualize=opt.visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=opt.line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if opt.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or opt.view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if opt.save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if opt.view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        
        
        
        #botsort
        frame_id = frame
        frame = im0s.copy()
        
        bboxes = []
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = [cls, *xywh, conf]
            bboxes.append(line)
        
        
        bbox_xywh = []
        for cls_id, x, y, w, h, conf in bboxes:
            obj = [
                int(max((x - w / 2) * width, 0)), int(max((y - h / 2) * height, 0)),
                int(min((x + w / 2) * width, width - 1)), int(min((y + h / 2) * height, height - 1)),
                conf.cpu(), 0
            ]
            bbox_xywh.append(obj)

        if bbox_xywh is not None:
            detections = np.array(bbox_xywh)

            # Run tracker
            online_targets = tracker.update(detections, frame)

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > opt.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    results.append(
                        f"{getattr(dataset, 'frame', 0) - 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
                    results_list.append([getattr(dataset, 'frame', 0) - 1, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], t.score])
            online_im = plot_tracking(
                    frame, online_tlwhs, online_ids, frame_id=frame_id, fps=fps)
        else:
            online_im = frame
        if save_img:
            vid_writer0.write(online_im)
            
    #botsort results
    if opt.save_txt:
        res_file = osp.join(save_folder, f"{p.stem}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
        

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *opt.imgsz)}' % t)
    if opt.save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if opt.save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if opt.update:
        strip_optimizer(opt.weights[0])  # opt.update model (to fix SourceChangeWarning)
              
    #easy case
    if not results_list:
        return
    df = pd.DataFrame(results_list)
    tids = set(df[1])
    obj_cnt = {}
    is_easycase = False
    for tid in tids:
        first_frame = df[df[1] == tid].iloc[0, :][2:6].tolist()
        last_frame = df[df[1] == tid].iloc[-1, :][2:6].tolist()
        y1 = first_frame[1] + first_frame[3] / 2
        y2 = last_frame[1] + last_frame[3] / 2
        if abs(y2 - y1) >= 0.17 * height:
            obj_cnt[tid] = y2 - y1
    if len(obj_cnt) == 1 and list(obj_cnt.values())[0] > 0 and len(tids) <= 8:
        is_easycase = True
        tid = list(obj_cnt.keys())[0]
    if is_easycase:
        os.makedirs(f'{tmpdir}/image_results', exist_ok=1)
        cap = cv2.VideoCapture(opt.source)
        num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        boxes = df[df[1] == tid].values.tolist()
        image_boxes = [boxes[np.linspace(0, len(boxes) - 1, 5, dtype=int)[i]] for i in range(5)]
        max_l = 0
        for i in range(0, num_frame):
            _, frame = cap.read()
            while image_boxes and i == int(image_boxes[0][0]):
                x, y, w, h = image_boxes[0][2:6]
                x0, y0 = int(x + w / 2), int(y + h / 2)
                l = int(max(w, h)) + 50
                x1, y1, x2, y2 = x0 - l // 2, y0 - l // 2, x0 + l // 2, y0 + l // 2
                if x1 < 0:
                    x1, x2 = 0, x2 - x1
                if y1 < 0:
                    y1, y2 = 0, y2 - y1
                if x2 > W:
                    x1, x2 = x1 - (x2 - W), W
                if y2 > H:
                    y1, y2 = y1 - (y2 - H), H
                if l > max_l:
                    max_l = l
                    max_image = frame[y1:y2, x1:x2, :].copy()
                image_boxes.pop(0)
        cv2.imwrite(f'{tmpdir}/image_results/{p.stem}.jpg', max_image)
        print(f'Results saved to {tmpdir}/image_results')
        print('This video is an EasyCase')
    else:
        print('This video is not an EasyCase')

def run(opt):
    # Load model
    opt.device = select_device(opt.device)
    model = DetectMultiBackend(opt.weights, device=opt.device, dnn=opt.dnn, data=opt.data, fp16=opt.half)
    is_dir = os.path.isdir(opt.source)
    if is_dir:
        files_last = []
        files = os.listdir(opt.source)
        files_diff = files
        while len(files_diff) != 0:
            files_last = files
            for file in tqdm(files_diff):
                is_file = Path(file).suffix[1:] in (IMG_FORMATS + VID_FORMATS) and file[0] != '.'
                if is_file:
                    opt1 = copy.deepcopy(opt)
                    opt1.source = opt.source + '/' + file
                    print(opt1.source)
                    run_by_video(model, opt1)
            files = os.listdir(opt.source) #检查源文件夹是否有更新
            files_diff = list(set(files) - set(files_last))
    else:
        run_by_video(model,opt)


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(opt)


if __name__ == "__main__":
    opt = parse_opt()
    
    
    
    main(opt)
    
