과학기술정보통신부 인공지능 개발자 양성과정 최종 팀프로젝트

# 사용자별 행동 인식 스마트 홈

### 👋 Introduction

<table>
    <tr>
        <th>프로젝트 명 </th>
        <th>사용자별 행동 인식 <br> 스마트 홈</th>
        <th>개발기간</th>
        <th>21.07.17 - 21.10.22</th>
    </tr>
    <tr>
        <th>프로젝트 성격</th>
        <th>API 통신, 인공지능, 웹 서버 </th>
        <th>개발인원</th>
        <th>팀 / 3명<br>
          (<a href="https://github.com/ropering">오명균</a>｜<a href="https://github.com/namyoungjun96">남영준</a>) <br>
          (<a href="https://github.com/yunyoung-Lee/">이윤영</a>)
      </th>
    </tr>
      <tr>
        <th>프로젝트 개요</th>
        <th>스마트 홈 서비스</th>
        <th>개발환경&nbsp;</th>
        <th>
            (AI 서버) <br> - Python (3.6.8)  <br> - Django (3.2.6) <br> 
            (IoT 서버) <br> - Node.js (LTS: 14.17.6)  <br> - RaspberryPi 4 <br>
            (Web 서버) <br> - Java (1.8) <br> - MariaDB <br>
        </th>
    </tr>
    <tr>
        <th>사용 도구</th>
        <th colspan="3">Pycharm, VSCode, Eclipse, Spring Tools 4</th>
    </tr>  
    <tr>
        <th>개발언어</th>
        <th colspan="3">Python, Java, Node.js, SQL, Html, CSS, Javascript</th>
    </tr>
    <tr>
        <th>사용기술</th>
        <th colspan="3" style="text-align:right;">
            <br> (Web 서버) MariaDB, API 통신 
            <br> (AI 서버) <br>
                - 얼굴 탐지(Ultra light), 인식(MobileFaceNet) <br>
                - 행동인식(Mediapipe Hands) <br>
                - API 통신(Django)
                - 실시간 스트리밍(Socket.IO)
            <br> (IoT 서버) 실시간 스트리밍(Socket.IO)
        </th>
    </tr>
</table>

### 📼 주제
▶ 카메라와 인공지능 모델을 이용한 사용자 별 손동작 인식 스마트 홈 구축 

### 🎈 구현 목적 <br>
▶ 코로나 19로 집안에서의 생활시간 증가 및 스마트 홈 시장 규모 증가<br>
▶ 사용자가 간편하게 손동작 만으로 각종 전자기기를 제어할 수 있도록 하기 위함<br><br>

### 📡 구현 기술 <br>
▶ (AI) 얼굴 탐지, 얼굴 인식, 손가락 인식 <br> (Ultra light face generic face detector, MobileFaceNet, Mediapipe Hands) <br>
▶ (Server) 웹 서버 구축 (Django, Spring, Node.js) <br>
▶ 서버 간 REST API <br> <br>

### 🎫 구현 역할 <br>
▶ 웹 서버 구현 및 REST API 구현 (Django) <br>
▶ 인공지능 모델링 (face detection, face recognition, finger recognition) <br>
▶ 프로젝트 기획 및 프로그래밍 <br> <br>

### 🧧 관련 자료 <br>
▶ https://github.com/ropering/User-specific-behavior-recognition-smart-home <br>
▶ https://github.com/yunyoung-Lee/SpringWebServer <br>
▶ https://github.com/namyoungjun96/IOTNodejs <br>

### 📆 일정
![image10](https://user-images.githubusercontent.com/50795314/138651337-7bb06b7b-0150-43da-8f9f-7a74ba21e281.png)

### 📚 아키텍처 설계
![image](https://user-images.githubusercontent.com/50795314/138651492-f9ca7953-9a3c-4129-9609-1702aa6b52e7.png)

### 📚 시스템 구조도
![image](https://user-images.githubusercontent.com/50795314/138651894-63084a76-a73a-447c-b0a1-3f7778433d0f.png)

### 📚 DB 설계
![image29](https://user-images.githubusercontent.com/50795314/138651766-5c228421-5a12-4e08-b86c-c87ee72e6918.png)

### 🖥️ 인공지능 모델
![image](https://user-images.githubusercontent.com/50795314/138652222-59859d8c-e875-4d91-a555-9365683f1bc1.png)
![image](https://user-images.githubusercontent.com/50795314/138652273-4bbd7475-eef7-4c60-bd9d-c654c1ed0ed2.png)
![image](https://user-images.githubusercontent.com/50795314/138652291-3b977c45-6be7-4666-a08e-3bcbafe3b440.png)
![image](https://user-images.githubusercontent.com/50795314/138652303-8b05814b-6efb-4e89-a889-704c26979511.png)
![image](https://user-images.githubusercontent.com/50795314/138652334-f084b178-8361-4fc8-8451-1253de16e9a0.png)

### 📉 프로젝트 관리
![image](https://user-images.githubusercontent.com/50795314/138652378-49346f68-1ad7-4bf1-a0bd-b347d4b73c63.png)

### 📑 Role & Member

<table>
    <tr>
        <th width="16%">업무 / 구성원</th>
        <th width="14%">오명균</th>
        <th width="14%">남영준</th>        
        <th width="14%">이윤영</th>        
    </tr>
    <tr>
        <th>프로젝트 기획</th>
        <th colspan="3"> <center>개요작성, 회의, 의견제안 </center> </th>
    </tr>
    <tr>
        <th>요구분석</th>
        <th colspan="3"> <center> 문서작성, 회의, 의견제안 </center> </th>
    </tr>
        <th>소스(코딩)</th>
        <th>
            <br>인공지능 서버 개발(Django), 
            <br>인공지능 모델링,
            <br>소스 취합 및 수정
        </th>
        <th>
            <br>프로젝트 총괄, 
            <br>IoT 서버 개발(Node.js),
            <br>소스 취합 및 수정
        </th>
        <th>
            <br>웹 서버 개발(Spring),
            <br>PPT 디자인  
            <br>소스 취합 및 수정
        </th>
    </tr>
</table>

### 📽 시연 영상 

[![시연영상](http://img.youtube.com/vi/-GmJ03eB_88/0.jpg)](https://youtu.be/-GmJ03eB_88?t=0s) 
<br> https://youtu.be/-GmJ03eB_88
