# ignore warnings
import warnings

import cv2
import numpy as np

warnings.filterwarnings(action='ignore')
# from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
# from django.views.decorators.http import require_POST # 파일전송
# from django.views.decorators.csrf import csrf_exempt # 파일전송
# from .. import FileManager as fm
import sys
# from .main import test
from . import test, FingerCounter, train
# sys.path.append(r"C:\\_workspace\\_python\\2021-3Q\\ai_server\\iot\\main") # Adds higher directory to python modules path.
# from main import test
# from main import train
# from main import FingerCounter
# video upload
# https://ibit.ly/D9uW
from .forms import Video_form
# from .models import Video
import glob
# remove files
import os
# CSRF verification failed. Request aborted
from django.views.decorators.csrf import csrf_exempt
# from django.views.decorators.csrf import csrf_protect
# video streaming
import socketio
import base64
# HTTP Library for Python
import requests

import time


# 얼굴인식 객체 생성
t = test.FaceRecognition()
sio = socketio.Client()

isa = False
webServer_url = "http://192.168.30.27:8080/ajax/FingerSendData/"
iotServer_url = 'http://192.168.30.29:3000'


@sio.on('liveStream') # 'liveStream'이라고 오는 event message를 수신한다
def message(data):
    # print('I received a message')
        # decodeit = open('streaming.jpg', 'wb')
        # decodeit.write(base64.b64decode((buffer)))
        # print('사진 저장')
        # decodeit.close()
        # 얼굴 인식
        global isa
        if isa == False:
            isa = True
            # try:
            #     buffer = data['buffer'] # base64 타입으로 변환된 이미지
            # except TypeError:
            #     print("스트리밍 오류")
            # else:
            # # 사진 저장
            #     with open('streaming.jpg', 'wb') as f:
            #         f.write(base64.b64decode(buffer))
            #         f.flush()
            #     print("이미지 저장 완료")
            buffer = data['buffer']
            # print(f"buffer is {buffer}")
            decoded = base64.b64decode(buffer)
            # print(f"decoded is {decoded}")
            npimg = np.fromstring(decoded, dtype=np.uint8)
            # print(npimg)
            # print(type(npimg))
            # print(f"npimg is {npimg}")
            global img
            # if not None:
            img = cv2.imdecode(npimg, 1) # imdecode : Reads an image from a buffer in memory
            # img 변수의 자료형이 numpy 배열 자료형이라면 진행
            if isinstance(img, np.ndarray):
                # print(img.shape)
                # print(type(img))
                                         # imread : Loads an image from a file
            # print(img)
            # while True:
            # cv2.imshow('img', img)
            # print(f"img is {img} \n type is {type(img)}")

                face_result = t.start_test(img)
                finger_result = FingerCounter.start_fingerRecognition(img)



                if face_result and finger_result:
                    # print("=" * 20)
                    # print(f"얼굴인식 결과 : {face_result}")
                    # print(f"행동인식 결과 : {finger_result}")
                    # 사용자별 행동 요청
                    global webServer_url
                    webServer_url += face_result + "/" + finger_result
                    print(webServer_url)
                    res = requests.post(webServer_url) # data={'face_result': face_result, 'finger_result': finger_result
                    webServer_url = "http://192.168.30.27:8080/ajax/FingerSendData/"
                    user_behavior = None
                    if res.status_code == 200:
                        # print("전송 성공 코드: 200")
                        user_behavior = res.text
                        # print(f"반환 결과 값: {user_behavior}")
                    else:
                        print(f"오류 발생. 오류코드: {res.status_code}")

                    # sio.emit(user_behavior)
                    # if str(finger_result) == '1':
                    #     sio.emit('onFinger', finger_result) #{'result': face_result}
                    #
                    # if str(finger_result) == '5':
                    #     sio.emit('offFinger', finger_result) #{'result': face_result}

            isa = False
        else:
            buffer = ""

# @sio.on('storeImage')
# def storeImage():
#     for i in range(1, 100):
#         # pass
#         #
#         cv2.imwrite(f'faces/training/{i}.jpg', img)
#         time.sleep(0.5)




    # @sio.on('liveStream')
    # def image(data):
    #     for i in range(1, 100):
    #         global isa
    #         if isa == False:
    #             isa = True
    #
    #             buffer = data['buffer']
    #             decoded = base64.b64decode(buffer)
    #             npimg = np.fromstring(decoded, dtype=np.uint8)
    #             img = cv2.imdecode(npimg, 1)
    #
    #             cv2.imwrite('faces/training/*.jpg', grayImg)



@sio.event
def connect():
    print("I'm connected! \n")
    # sio.emit('start-stream')
@sio.event
def connect_error(data):
    print("The connection failed! \n")
@sio.event
def disconnect():
    print("I'm disconnected! \n")

try:
    sio.connect(iotServer_url)
except Exception as e: # Connection refused 처리
    print(f"에러: {e}")
else:
    sio.emit('start-stream')
    print("스트리밍 요청완료")


# 학습시 upload() 실행 -> face_train() 실행
@csrf_exempt
def add_face(request, id): # Web Server -> AI Server
    # id 폴더 생성
    train.create_folder(r"C:\_workspace\_python\2021-3Q\ai_server\iot\faces\training/", id) # ./iot/faces/training/
    # 영상 저장, 영상을 id 폴더 내로 이동
    if request.method == "POST":
        form = Video_form(data=request.POST, files=request.FILES)
        if form.is_valid():
            form.save()
            print("파일 저장 완료")
            video_name = glob.glob('faces/training/*.mp4')[0].split('\\')[-1]  # training 폴더에 있는 mp4 파일명
            train.move_file(video_name, id)
            print("파일 이동 완료")
            # 학습 시작
            print("학습 시작")
            train.start_train()
        else:
            print("파일 크기가 50MB가 넘습니다")
    else:
        print("영상 저장 실패 (POST 방식이 아닙니다)")

    print("학습 시작!!")
    print("="*10)

    if train.start_train() == 1:
        print("학습 성공")
        return HttpResponse(1)
    else:
        print("학습 실패")
        return HttpResponse(0)

@csrf_exempt
def get_recognition(request):
    # 영상 저장, 영상을 id 폴더 내로 이동
    # print(f"request.FILES : {request.FILES}")
    # print(f"request.POST : {request.POST}")
    if request.method == "POST":
        form = Video_form(data=request.POST, files=request.FILES)
        if form.is_valid():
            form.save()
            print("파일 저장 완료")

            img_name = glob.glob('faces/training/*.jpg')[0].split('\\')[-1]  # training 폴더에 있는 jpg 파일명

            face_result = t.start_test(f"faces/training/{img_name}")
            finger_result = FingerCounter.start_fingerRecognition(f"faces/training/{img_name}")

            os.remove(f"faces/training/{img_name}")  # 인식에 활용한 영상 파일 삭제
            return JsonResponse({'face_result': face_result, 'finger_result': finger_result})
        else:
            print("form이 유효하지 않습니다")
        # else:
        #     print("파일 크기가 50MB가 넘습니다")
        #     return HttpResponse("파일 크기가 50MB가 넘습니다")

    else:
        print("영상 저장 실패 (POST 방식이 아닙니다)")
        return HttpResponse("영상이 저장되지 않았습니다")



















@csrf_exempt
def recognition_behavior(request):
    if request.method == "POST":
        form = Video_form(data=request.POST, files=request.FILES)
        if form.is_valid():
            form.save()
            print("파일 저장 완료")

            video_name = glob.glob('faces/training/*.mp4')[0].split('\\')[-1]  # training 폴더에 있는 mp4 파일명
            result = FingerCounter.start_fingerRecognition(f"faces/training/{video_name}")
            os.remove(f"faces/training/{video_name}")# 인식에 활용한 영상 파일 삭제
            return HttpResponse(result) # 0 입력 시 카메라 캠 사용
    else:
        print("인식 실패")


@csrf_exempt
def recognition_face(request):
    # if request.method == "POST":
    #     form = Video_form(data=request.POST, files=request.FILES)
    #     if form.is_valid():
    #         form.save()
    #         print("파일 저장 완료")
    #
    video_name = glob.glob('faces/training/*.mp4')[0].split('\\')[-1]  # training 폴더에 있는 mp4 파일명
    t = test.FaceRecognition()
    result = t.start_test(f"faces/training/{video_name}")
    # result = t.start_test("./hello_level.jpeg")
            # os.remove(f"faces/training/{video_name}")# 인식에 활용한 영상 파일 삭제
    return HttpResponse(result)# 0 입력 시 카메라 캠 사용
    # else:
    #     print("인식 실패")


@csrf_exempt
def upload(request, id):
    # id 폴더 생성
    train.create_folder("./faces/training/", id)
    # 영상 저장, 영상을 id 폴더 내로 이동
    # print(f"request.FILES : {request.FILES}")
    # print(f"request.POST : {request.POST}")
    if request.method == "POST":
        form = Video_form(data=request.POST, files=request.FILES)
        if form.is_valid():
            form.save()
            print("파일 저장 완료")
            # 파일 이동
            video_name = glob.glob('faces/training/*.mp4')[0].split('\\')[-1] # training 폴더에 있는 mp4 파일명
            train.move_file(video_name, id)
            return HttpResponse("uploaded successfully")
        else:
            print("파일 크기가 50MB가 넘습니다")
            return HttpResponse("파일 크기가 50MB가 넘습니다")
    else:
        print("영상 저장 실패 (POST 방식이 아닙니다)")
        return HttpResponse("영상이 저장되지 않았습니다")

def face_train(request):
    # t = test.FaceRecognition()
    # train.move_file("nam.mp4", id="3")
    return HttpResponse(train.start_train()) # return "학습 완료!"
    # train.start_train()
    # return HttpResponse(FingerCounter.start_fingerRecognition()) # 손가락 반환까지 3초 소요
    # name = "hello"
    # train.create_folder("./faces/training/" + name) # 해당 경로에 폴더 생성
    # train.store_video("nam.mp4", "id")


def face_recognition(request):
    t = test.FaceRecognition()
    return HttpResponse(t.start_test("C:/omg.mp4")) # 0 입력 시 카메라 캠 사용


def finger_recognition(request):
    return HttpResponse(FingerCounter.start_fingerRecognition("C:/finger_4.mp4")) # 0 입력 시 카메라 캠 사용


def index(request, id):
    return JsonResponse({'answer': id})
    # return HttpResponse(f"index 함수 실행 {id}")

