import argparse
import cv2
import numpy as np
import onnxruntime as ort

connect_list = [
        [0, 1],  # 00:鼻(nose) -> 01:左目(left eye)
        [0, 2],  # 00:鼻(nose) -> 02:右目(right eye)
        [1, 3],  # 01:左目(left eye) -> 03:左耳(left ear)
        [2, 4],  # 02:右目(right eye) -> 04:右耳(right ear)
        [3, 5],  # 03:左耳(left ear) -> 05:左肩(left shoulder)
        [4, 6],  # 04:右耳(right ear) -> 06:右肩(right shoulder)
        [5, 6],  # 05:左肩(left shoulder) -> 06:右肩(right shoulder)
        [5, 7],  # 05:左肩(left shoulder)  -> 07:左肘(left elbow)
        [7, 9],  # 07:左肘(left elbow) -> 09:左手首(left wrist)
        [6, 8],  # 06:右肩(right shoulder) -> 08:右肘(right elbow)
        [8, 10],  # 08:右肘(right elbow) -> 10:右手首(right wrist)
        [5, 11],  # 05:左肩(left shoulder) -> 11:左腰(left waist)
        [6, 12],  # 06:右肩(right shoulder) -> 12:右腰(right waist)
        [11, 12],  # 11:左腰(left waist) -> 12:右腰(right waist)
        [11, 13],  # 11:左腰(left waist) -> 13:左膝(left knee)
        [13, 15],  # 13:左膝(left knee) -> 15:左足首(left ankle)
        [12, 14],  # 12:右腰(right waist) -> 14:右膝(right knee),
        [14, 16],  # 14:右膝(right knee) -> 16:右足首(right ankle)
    ]

class E2Pose():
    def __init__(self, model_path, confThreshold=0.5):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, so)
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])

        self.confThreshold = confThreshold
    def draw_pred(self, frame, ret):
        for result in ret:
            keypoint_list = list(result['keypoints'][:, :2].astype(np.int32))

            for keypoint in keypoint_list:
                cx = keypoint[0]
                cy = keypoint[1]
                if cx > 0 and cy > 0:
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            for connect in connect_list:
                cx1 = keypoint_list[connect[0]][0]
                cy1 = keypoint_list[connect[0]][1]
                cx2 = keypoint_list[connect[1]][0]
                cy2 = keypoint_list[connect[1]][1]
                if cx1 > 0 and cy1 > 0 and cx2 > 0 and cy2 > 0:
                    cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        return frame

    def detect(self, srcimg):
        image_height, image_width = srcimg.shape[:2]
        img = cv2.resize(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB), (self.input_width, self.input_height))
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # Inference
        results = self.session.run(None, {self.input_name: img})
        # Post process
        kpt, pv = results
        pv = np.reshape(pv[0], [-1])
        kpt = kpt[0][pv >= self.confThreshold]
        kpt[:, :, -1] *= image_height
        kpt[:, :, -2] *= image_width
        kpt[:, :, -3] *= 2
        ret = []
        for human in kpt:
            mask = np.stack([(human[:, 0] >= self.confThreshold).astype(np.float32)], axis=-1)
            human *= mask
            human = np.stack([human[:, _ii] for _ii in [1, 2, 0]], axis=-1)
            ret.append({'keypoints': human, 'category_id': 1})
        return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", type=str, default='weights/e2epose_resnet50_1x3x512x512.onnx', help="model path")
    parser.add_argument("--imgpath", type=str, default='images/person.jpg', help="image path")
    parser.add_argument("--confThreshold", default=0.5, type=float, help='class confidence')
    args = parser.parse_args()

    net = E2Pose(args.modelpath, confThreshold=args.confThreshold)
    if args.imgpath.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        srcimg = cv2.imread(args.imgpath)
        results = net.detect(srcimg)
        srcimg = net.draw_pred(srcimg, results)

        winName = 'Deep learning object detection in ONNXRuntime'
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.imshow(winName, srcimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        videopath = args.imgpath
        cap = cv2.VideoCapture(videopath)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        winName = 'Deep learning object detection in ONNXRuntime'

        frame_id = 0
        while True:
            # frame_id +=1
            # if frame_id%2!=0:
            #     continue
            ret, frame = cap.read()
            if not ret:
                break

            results = net.detect(frame)
            drawimg = np.zeros_like(frame)
            drawimg = net.draw_pred(drawimg, results)

            cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
            cv2.imshow(winName, np.hstack((frame, drawimg)))
            cv2.waitKey(1)
        cv2.destroyAllWindows()