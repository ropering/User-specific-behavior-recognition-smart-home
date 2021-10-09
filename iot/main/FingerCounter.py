'''
https://www.youtube.com/watch?v=p5Z_GGRCI5s
https://www.computervision.zone/lessons/code-files-13/
손가락 모양에 따른 사진 및 숫자 판단
문제
- 손바닥 위치에서는 잘 인식
- 기울기에 따라, 손등에 따라 제대로 된 인식 불가능

'''
import cv2
import time
import os
from . import HandTrackingModule as htm

import time

def start_fingerRecognition(img):
    wCam, hCam = 640, 480
    set_ = set()
    count_ = 0

    # cap = cv2.VideoCapture(file_path)
    # cap.set(3, wCam)
    # cap.set(4, hCam)

    # folderPath = "FingerImages"
    # myList = os.listdir(folderPath)
    # print(myList)
    # overlayList = []
    # for imPath in myList:
    #     image = cv2.imread(f'{folderPath}/{imPath}')
    #     # print(f'{folderPath}/{imPath}')
    #     overlayList.append(image)
    #
    # print(len(overlayList))
    pTime = 0

    detector = htm.handDetector(detectionCon=0.75)
    tipIds = [4, 8, 12, 16, 20]

    while True: # 이 부분에서 꽤 많은 반복(delay)
        # img = cv2.imread(file_path)
        # success, img = cap.read()
        # success
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        # print(lmList)

        if len(lmList) != 0: # 손이 감지 된다면
            fingers = []

            # Thumb
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # print(fingers)
            cv2.destroyAllWindows() # cvtError 해결 (이미지 못불러오는 문제를 이 소스로 해결)
            # print(fingers)
            totalFingers = fingers.count(1)
            if totalFingers == 0: # 0이 자주 찍혀서 0 제거
                break
            else:
                return str(totalFingers)
            # print(totalFingers)
        else:
            return None

            '''손모양 정확도 개선 알고리즘'''
'''
            set_.add(fingers.count(1)) # 현재 손 모양 추가
            count_ += 1

            if len(set_) > 1:
                set_.clear()
                count_ = 0

            # cv2.imshow('img', img)
            # time.sleep(1)
            if count_ > 4: # 해당 손 모양이 1초 동안 유지된다면 (FPS가 평균 30이기때문에 1초)
                count_ = 0
                return set_.pop() # 터미널에 손 모양 표시
        else:
            return "손이 감지되지 않습니다"
'''

            # h, w, c = overlayList[totalFingers - 1].shape
            # img[0:h, 0:w] = overlayList[totalFingers - 1]

            # cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            # cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
            #             10, (255, 0, 0), 25)

        # cTime = time.time()
        # fps = 1 / (cTime - pTime)
        # pTime = cTime

        # cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
        #             3, (255, 0, 0), 3)

        # cv2.imshow("Image", img)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # cap.release()
    # cv2.destroyAllWindows()



