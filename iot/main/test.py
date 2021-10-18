# -*- coding: utf-8 -*-
# @Author: fyr91
# @Date:   2019-10-22 15:05:15
# @Last Modified by:   User
# @Last Modified time: 2019-10-30 22:06:32
import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
import tensorflow as tf
import pickle
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
import time
'''
1. ultra light face detector 사용
2. face landmarks 검출 
3. train을 통해 학습 및 저장시킨 모델(.pkl 확장자) 불러오기 

- MFN : Multiple Face Network
'''

class FaceRecognition(object): # 싱글턴 패턴 적용
    """
    - 하나의 객체만 만들 수 있다
    - 공유된 리소스에 대한 동시 접근 제어 가능
    - 전역 객체를 제공한다
    """
    def first(self):
        print("객체를 생성합니다")
        self.onnx_path = '../models/ultra_light/ultra_light_models/ultra_light_640.onnx'
        self.onnx_model = onnx.load(self.onnx_path)
        self.predictor = prepare(self.onnx_model)
        self.ort_session = ort.InferenceSession(self.onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        # ./models/facial_landmarks/shape_predictor_5_face_landmarks.dat
        self.shape_predictor = dlib.shape_predictor("C:/_workspace/_python/2021-3Q/ai_server/models/facial_landmarks/shape_predictor_5_face_landmarks.dat")
        self.fa = face_utils.facealigner.FaceAligner(self.shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

        self.threshold = 0.63

        # load distance
        with open("embeddings/embeddings.pkl", "rb") as f:  # 학습시 생성되는 모델
            (self.saved_embeds, self.names) = pickle.load(f)

        tf.Graph().as_default()

        self.sess = tf.Session()
        saver = tf.train.import_meta_graph('../models/mfn/m1/mfn.ckpt.meta')
        saver.restore(self.sess, '../models/mfn/m1/mfn.ckpt')

        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]
        print("객체가 생성되었습니다")

    def __new__(cls):
        # print("생성자 함수 실행")
        if not hasattr(cls, 'instance'):
            print("만들어진 객체가 없으므로 객체를 생성합니다")
            cls.instance = super().__new__(cls)
            cls.instance.first()
            return cls.instance
        # print("객체가 이미 존재합니다 이미 만들어진 객체를 반환합니다")
        else:
            return cls.instance

    def area_of(self, left_top, right_bottom):
        """
        Compute the areas of rectangles given two corners.
        Args:
            left_top (N, 2): left top corner.
            right_bottom (N, 2): right bottom corner.
        Returns:
            area (N): return the area.
        """
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

    def iou_of(self, boxes0, boxes1, eps=1e-5):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Args:
            boxes0 (N, 4): ground truth boxes.
            boxes1 (N or 1, 4): predicted boxes.
            eps: a small number to avoid 0 as denominator.
        Returns:
            iou (N): IoU values.
        """
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 = self.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def hard_nms(self, box_scores, iou_threshold, top_k=-1, candidate_size=200):
        """
        Perform hard non-maximum-supression to filter out boxes with iou greater
        than threshold
        Args:
            box_scores (N, 5): boxes in corner-form and probabilities.
            iou_threshold: intersection over union threshold.
            top_k: keep top_k results. If k <= 0, keep all the results.
            candidate_size: only consider the candidates with the highest scores.
        Returns:
            picked: a list of indexes of the kept boxes
        """
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        indexes = np.argsort(scores)
        indexes = indexes[-candidate_size:]
        while len(indexes) > 0:
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = self.iou_of(
                rest_boxes,
                np.expand_dims(current_box, axis=0),
            )
            indexes = indexes[iou <= iou_threshold]

        return box_scores[picked, :]

    def predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
        """
        Select boxes that contain human faces
        Args:
            width: original image width
            height: original image height
            confidences (N, 2): confidence array
            boxes (N, 4): boxes array in corner-form
            iou_threshold: intersection over union threshold.
            top_k: keep top_k results. If k <= 0, keep all the results.
        Returns:
            boxes (k, 4): an array of boxes kept
            labels (k): an array of labels for each boxes kept
            probs (k): an array of probabilities for each boxes being in corresponding labels
        """
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = self.hard_nms(box_probs,
               iou_threshold=iou_threshold,
               top_k=top_k,
               )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    def start_test(self, frame):
        # start1 = time.time()
        # '''1'''
        #
        #
        # onnx_path = '../models/ultra_light/ultra_light_models/ultra_light_640.onnx'
        # onnx_model = onnx.load(onnx_path)
        # predictor = prepare(onnx_model)
        # ort_session = ort.InferenceSession(onnx_path)
        # input_name = ort_session.get_inputs()[0].name
        #
        # shape_predictor = dlib.shape_predictor('../models/facial_landmarks/shape_predictor_5_face_landmarks.dat')
        # fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))
        #
        # threshold = 0.63
        #
        # print(f"1: {time.time() - start1}")
        # start2 = time.time()
        # '''2'''
        #
        # # load distance
        # with open("embeddings/embeddings.pkl", "rb") as f: # 학습시 생성되는 모델
        #     (saved_embeds, names) = pickle.load(f)
        # print(f"2: {time.time() - start2}")
        #
        #
        # print(f"2: {time.time() - start2}")
        # start3 = time.time()
        # '''3'''

        # with tf.Graph().as_default():
        #
        #     with tf.Session() as sess:
                #
                # saver = tf.train.import_meta_graph('../models/mfn/m1/mfn.ckpt.meta')
                # saver.restore(sess, '../models/mfn/m1/mfn.ckpt')
                #
                # images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                # phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                # embedding_size = embeddings.get_shape()[1]
                #
                # print(f"3: {time.time() - start3}")
                # start4 = time.time()
                # '''4'''
                # 여기부터 카메라
                # video_capture = cv2.VideoCapture(file_path)
        while True:
            # fps = video_capture.get(cv2.CAP_PROP_FPS)
            # ret, frame = video_capture.read()

            # frame = cv2.imread(file_path)

            # preprocess faces
            h, w, _ = frame.shape
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img_mean = np.array([127, 127, 127])
            img = (img - img_mean) / 128
            img = np.transpose(img, [2, 0, 1])
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.float32)

            # detect faces (img : 캠에서 캡처한 얼굴 <-> 모델과 비교)
            confidences, boxes = self.ort_session.run(None, {self.input_name: img})
            boxes, labels, probs = self.predict(w, h, confidences, boxes, 0.7)

            # locate faces
            # 얼굴 부분만 잘라내서 append to faces
            # box : 얼굴을 둘러싼 네모 모양
            faces = []
            boxes[boxes<0] = 0
            for i in range(boxes.shape[0]):
                box = boxes[i, :]
                x1, y1, x2, y2 = box
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                aligned_face = self.fa.align(frame, gray, dlib.rectangle(left = int(x1), top=int(y1), right=int(x2), bottom=int(y2)))
                aligned_face = cv2.resize(aligned_face, (112,112))

                aligned_face = aligned_face - 127.5
                aligned_face = aligned_face * 0.0078125

                faces.append(aligned_face)

            # face embedding : 얼굴로부터 추출한 특징을 나타내는 벡터
            # 얼굴이 탐지되었다면
            if len(faces)>0:
                predictions = []

                faces = np.array(faces)
                feed_dict = {self.images_placeholder: faces, self.phase_train_placeholder: False}
                embeds = self.sess.run(self.embeddings, feed_dict=feed_dict)

                # prediciton using distance
                # trainin 때 학습시킨 모델과 비교해서 차이를 변수에 저장
                for embedding in embeds:
                    # saved_embeds : 학습시킨 embedding.pkl 파일값
                    diff = np.subtract(self.saved_embeds, embedding)
                    dist = np.sum(np.square(diff), 1)
                    idx = np.argmin(dist)
                    if dist[idx] < self.threshold:
                        predictions.append(self.names[idx])
                    else:
                        return None
                        # predictions.append("unknown")
                # draw
                for i in range(boxes.shape[0]):
                    box = boxes[i, :]

                    text = f"{predictions[i]}"

                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80,18,236), 2)
                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80,18,236), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255), 1) # putText 내부에 왜 구현부분이 없지?
                    # print(f"얼굴인식 완료 : {text}")
                    cv2.destroyAllWindows()
                    return str(text)
            else:
                # print("no face")
                return None

            # cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        # video_capture.release()
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    pass
    # start_test()


