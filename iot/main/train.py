# -*- coding: utf-8 -*-
# @Author: fyr91
# @Date:   2019-10-22 15:05:15
# @Last Modified by:   User
# @Last Modified time: 2019-10-30 22:04:25
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
import shutil # 파일 이동 라이브러리


def create_folder(path: str, id: str): # 자료형 변환
    try:
        os.mkdir(path + id)
    except FileExistsError:
        print("폴더가 이미 존재합니다! \n")
        return "폴더가 이미 존재합니다! \n"
    except Exception as e:
        print("오류 발생 \n" + str(e) + "\n")
        return "오류 발생" + str(e)


def move_file(file : str, id : str):
    """ C드라이브에 위치한 파일을 -> id 폴더 내로 이동 """
    file_name = file
    src_path = r"C:\_workspace\_python\2021-3Q\ai_server\iot\faces\training/" # raw 문자열 : escape 문자 적용되지 않고 그대로 출력된다
    dest_path = f"C:\\_workspace\\_python\\2021-3Q\\ai_server\\iot\\faces\\training/{id}/"
    shutil.move(src_path + file_name, dest_path + file_name)
    print("파일 이동 완료")

# def store_video(file : str, id: str):
#     print("시작")
#     import cv2
#     # cap = cv2.VideoCapture("iot/main/" + file)
#     cap = cv2.VideoCapture("iot/main/" + "nam.mp4")
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 또는 cap.get(3)
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 cap.get(4)
#     fps = cap.get(cv2.CAP_PROP_FPS) # 또는 cap.get(5)
#     fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # DIVX 코덱 적용
#
#     while True:
#         ret, frame = cap.read()
#         out = cv2.VideoWriter('DIVX.mp4', int(fcc), int(fps), (int(width), int(height)))  # 비디오 저장을 위한 객체를 생성한다. # cv2.VideoWriter(저장 위치, 코덱, 프레임, (가로, 세로))
#         out.write(frame)
#         cv2.imshow('frame', frame)
#
#         if cv2.waitKey(1) & 0xFF == 'q':
#             break
#     # https://ibit.ly/iDsr
# 주어진 두 개의 모서리 점으로부터 사각형의 넓이 계산


def area_of(left_top, right_bottom): # 사각형 영역 계산
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


# 특정한 데이터셋에서 object detector의 정확도를 측정하기 위해서 사용된다
def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2]) # 교집합 영역
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom) # 교집합 영역 넓이
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:]) # 합집합 영역 넓이
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps) # 교집합 영역 넓이 / 합집합 영역 넓이


# 임계값보다 큰 iou가 있는 boxes를 걸러내기 위해 hard non-maximum-suppression을 수행
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
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
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


# Select boxes that contain human faces
def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
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
        box_probs = hard_nms(box_probs,
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


def start_train():
    '''
    face detect
    얼굴을 찾아내는 것 : ultra light face detector 사용 (onnx)
    '''
    onnx_path = '../models/ultra_light/ultra_light_models/ultra_light_640.onnx'
    onnx_model = onnx.load(onnx_path)
    predictor = prepare(onnx_model)
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    shape_predictor = dlib.shape_predictor(r'../models/facial_landmarks/shape_predictor_5_face_landmarks.dat')
    fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

    # 얼굴 인식할 파일 경로
    TRAINING_BASE = 'faces/training/'

    dirs = os.listdir(TRAINING_BASE)
    images = []
    names = []

    for label in dirs: # dirs : 영상이 들어있는 폴더 목록 (폴더 명으로 label을 붙인다 (얼굴인식))
        for i, fn in enumerate(os.listdir(os.path.join(TRAINING_BASE, label))): # 폴더 접근
            print(f"start collecting faces from {label}'s data")
            cap = cv2.VideoCapture(os.path.join(TRAINING_BASE, label, fn)) # 동영상 파일 접근
            frame_count = 0
            while True:
                # 학습할 영상 한 프레임씩 가져오기
                ret, raw_img = cap.read()
                # 프레임 마다 전처리 -> 얼굴 정확도 산출
                if raw_img is not None: # frame_count % 5 == 0 and
                    # 프레임 전처리
                    h, w, _ = raw_img.shape
                    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (640, 480))
                    img_mean = np.array([127, 127, 127])
                    img = (img - img_mean) / 128
                    img = np.transpose(img, [2, 0, 1])
                    img = np.expand_dims(img, axis=0)
                    img = img.astype(np.float32)

                    # 얼굴 정확도 산출
                    confidences, boxes = ort_session.run(None, {input_name: img})
                    boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

                    # if face detected
                    if boxes.shape[0] > 0:
                        x1, y1, x2, y2 = boxes[0,:]
                        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
                        aligned_face = fa.align(raw_img, gray, dlib.rectangle(left=int(x1), top=int(y1), right=int(x2), bottom=int(y2)))
                        aligned_face = cv2.resize(aligned_face, (112, 112))
                        # 해당 경로에 전처리된 이미지 저장
                        cv2.imwrite(f'./faces/tmp/{label}_{frame_count}.jpg', aligned_face)
                        # f'../faces/tmp/{label}_{frame_count}.jpg'
                        aligned_face = aligned_face - 127.5
                        aligned_face = aligned_face * 0.0078125
                        images.append(aligned_face)
                        names.append(label)

                frame_count += 1
                if frame_count == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    break
    # face training
    with tf.Graph().as_default():
        with tf.Session() as sess:
            print("loading checkpoint ...")
            saver = tf.train.import_meta_graph('../models/mfn/m1/mfn.ckpt.meta')
            saver.restore(sess, '../models/mfn/m1/mfn.ckpt')

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # images : 얼굴만 들어있는 이미지 목록
            feed_dict = {images_placeholder: images, phase_train_placeholder:False }
            # Tensorflow 연산 수행
            embeds = sess.run(embeddings, feed_dict=feed_dict)
            # 연산 결과 덮어씌우기
            with open("embeddings/embeddings.pkl", "wb") as f:
                pickle.dump((embeds, names), f)
            print("얼굴 학습 Done!")
            return 1

if __name__ == '__main__':
    # create_folder("../faces/training/", "3") # training.py 위치에서 실행했을 때 (서버에서 돌릴 때는 경로를 바꿔줘야함)
    # move_file("nam.mp4", 3)
    # create_folder("C:/_workspace/_python/2021-3Q/iot_server/iot/faces/training/", id="3")
    start_train()
    # move_file("nam.mp4", create_folder("C:/_workspace/_python/2021 - 3Q/iot_server/iot/faces/training/", id="3"))

    # print("시작")
    # import cv2
    #
    # # cap = cv2.VideoCapture("iot/main/" + file)
    # cap = cv2.VideoCapture("iot/main/" + "nam.mp4")
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 또는 cap.get(3)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 또는 cap.get(4)
    # fps = cap.get(cv2.CAP_PROP_FPS)  # 또는 cap.get(5)
    # fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # DIVX 코덱 적용
    #
    # while True:
    #     ret, frame = cap.read()
    #     out = cv2.VideoWriter('DIVX.mp4', int(fcc), int(fps), (int(width), int(height)))  # 비디오 저장을 위한 객체를 생성한다. # cv2.VideoWriter(저장 위치, 코덱, 프레임, (가로, 세로))
    #     out.write(frame)
    #     cv2.imshow('frame', frame)
    #     # row, column, depth = frame.shape
    #     if cv2.waitKey(1) & 0xFF == 'q':
    #         break