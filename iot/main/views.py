# ignore warnings
import warnings
warnings.filterwarnings(action='ignore')
import cv2
import numpy as np
from django.http import HttpResponse, JsonResponse
from . import test, FingerCounter, train
# sys.path.append(r"C:\\_workspace\\_python\\2021-3Q\\ai_server\\iot\\main") # Adds higher directory to python modules path.
from .forms import Video_form
import glob # find file name
import os # remove files
from django.views.decorators.csrf import csrf_exempt # CSRF verification failed. Request aborted
# video streaming
import socketio
import base64
# HTTP Library for Python
import requests

import re
import time


sio = socketio.Client()
# variable to store images
count = 1

isa = False
images = [] # store images for train
size = ()
webServer_url = "http://192.168.30.27:8080/ajax/FingerSendData/"
iotServer_url = "http://192.168.30.29:3000/"
# 'http://192.168.30.29:3000'
'''
@sio.on('liveStream') # 'liveStream'이라고 오는 event message를 수신한다
def message(data):
    global isa
    if isa == False:
        isa = True
        # 얼굴인식 객체 생성
        t = test.FaceRecognition()
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
        decoded = base64.b64decode(buffer)
        npimg = np.fromstring(decoded, dtype=np.uint8)
        # sio.emit('storeImage')
        # (-215:Assertionfailed) !buf.empty() in function 'cv::imdecode_'에러 처리
        try:
            img = cv2.imdecode(npimg, 1) # imdecode : Reads an image from a buffer in memory
            # print(img.shape)
        except Exception as e:
            print(f"오류 {e}")
        else:
            # start face recognition, hehavior recognition
            face_result = t.start_test(img)
            finger_result = FingerCounter.start_fingerRecognition(img)
            # print("="*10)
            # print(face_result)
            # print(finger_result)

            # if face and behavior is recognized
            if face_result and finger_result:
                # request to Web Server
                global webServer_url
                webServer_url += face_result + "/" + finger_result
                print(webServer_url)
                try:
                    res = requests.post(webServer_url)
                except Exception as e:
                    print(f"ERROR: {e}")
                # reset URL
                webServer_url = webServer_url[: webServer_url.find("Data/") + 5]
                # if res.status_code == 200:
                #     # print("전송 성공 코드: 200")
                # else:
                #     print(f"오류 발생. 오류코드: {res.status_code}")

                # sio.emit(user_behavior)
                # if str(finger_result) == '1':
                #     sio.emit('onFinger', finger_result) #{'result': face_result}
                #
                # if str(finger_result) == '5':
                #     sio.emit('offFinger', finger_result) #{'result': face_result}
        finally:
            isa = False
    else:
        buffer = ""

'''
# @sio.on('sendImage')
# def store_image(data):
#     print("이미지 저장 시작")
#     buffer = data['buffer']
#     decoded = base64.b64decode(buffer)
#     npimg = np.fromstring(decoded, dtype=np.uint8)
#     # (-215:Assertionfailed) !buf.empty() in function 'cv::imdecode_'에러 처리
#     try:
#         img = cv2.imdecode(npimg, 1)  # imdecode : Reads an image from a buffer in memory
#     except Exception as e:
#         print(f"오류 {e}")
#     else:
#         images.append(img)
#         print("이미지 저장 완료")
#         print(f"img len : {images}")


    # print("이미지 저장 시작")
    # id = ""
    # id = id.zfill(4) # 4자리 숫자로 채움 나머지는 0
    # # id값을 받을 수 있다면 파일명을 "id_x.jpg"이렇게 저장가능 -> 이미지를 삭제할 필요가 없어진다
    # buffer = data['buffer']
    # decoded = base64.b64decode(buffer)
    # npimg = np.fromstring(decoded, dtype=np.uint8)
    # # (-215:Assertionfailed) !buf.empty() in function 'cv::imdecode_'에러 처리
    # try:
    #     img = cv2.imdecode(npimg, 1)  # imdecode : Reads an image from a buffer in memory
    # except Exception as e:
    #     print(e)
    # else:
    #     cv2.imwrite(f'./faces/training/{id}_{count}', img)
    #     print("이미지 저장 완료")


# @sio.on('convertToVideo')
# def convert_to_video(data):
#     print("영상 저장 시작")
#     # global images
#     # global size
#     id = data['id']
#     print(id)
    # 학습완료를 알리기 위해 1 보내기 또는 0
    # train.create_folder(r"C:\_workspace\_python\2021-3Q\ai_server\iot\faces\training/", id)
    # path_out = f"faces/training/{id}"
    # fps = 30
    # frame_array = []
    #
    # for img in images:
    #     img = cv2.imread(img)
    #     height, width, layers = img.shape
    #     size = (width, height)
    #     frame_array.append(img)
    #
    #     print(f"\n\n\n\n{size}")
    # out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    # for i in range(len(frame_array)):
    #     out.write(frame_array[i])
    # out.release()
    # print("영상 저장 완료")
    # images = []


    # print("영상 저장 시작")
    # global size
    # id = ""
    # path = "faces/training/"
    # paths = [os.path.join(path, i) for i in os.listdir("iot/faces/training") if re.search(f"{id}_\d", i)] # bono_\d\d\d\d.jpg"
    #
    # train.create_folder(r"C:\_workspace\_python\2021-3Q\ai_server\iot\faces\training/", id)
    # path_out = f"faces/training/{id}"
    # fps = 30
    # frame_array = []
    #
    # for path in paths:
    #     img = cv2.imread(path)
    #     height, width, layers = img.shape
    #     size = (width, height)
    #     frame_array.append(img)
    # out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    # for i in range(len(frame_array)):
    #     out.write(frame_array[i])
    # out.release()
    # print("영상 저장 완료")

    # print("학습 시작")
    # train.start_train()


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
    # for _ in range(10):
    #     print("이미지 요청")
    #     sio.emit('storeImage')
    #     time.sleep(0.5)
    # sio.emit('convertToVideo')


# 학습시 upload() 실행 -> face_train() 실행

@csrf_exempt
def add_face(request, id): # Web Server -> AI Server
    # request to iot server & save video
    with requests.post(iotServer_url + "request", stream=True) as r:
        r.raise_for_status()
        with open(f'faces/training/{id}.mp4', 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    # id 폴더 생성
    train.create_folder(r"C:\_workspace\_python\2021-3Q\ai_server\iot\faces\training/", id) # ./iot/faces/training/
    # 영상 저장, 영상을 id 폴더 내로 이동
    form.save()
    print("파일 저장 완료")
    video_name = glob.glob('faces/training/*.mp4')[0].split('\\')[-1]  # training 폴더에 있는 mp4 파일명
    train.move_file(video_name, id)
    print("파일 이동 완료")
    # 학습 시작
    print("학습 시작")
    # if train.start_train() == 1:
    #     print("학습 성공")
    #     return JsonResponse({'state': 1})
    # else:
    #     print("학습 실패")
    #     return HttpResponse({'state': 0})



@csrf_exempt
def add_face(_, id):
    with requests.post(iotServer_url + "request", stream=True) as r:
        r.raise_for_status()
        with open(f'faces/training/{id}.mp4', 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return HttpResponse(1)





















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
    # result = t.start_test("./hello_level.jpg")
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

