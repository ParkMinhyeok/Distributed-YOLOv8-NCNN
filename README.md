# 라즈베리파이에서 NCNN을 활용한 YOLOv8 추론



이 저장소는 NCNN 추론 엔진을 사용하여 라즈베리파이에서 YOLOv8 실시간 객체 탐지를 구현하기 위한 단계별 가이드를 제공합니다. 모든 과정은 최고의 성능을 위해 C++ 환경에서 실행됩니다.

---
##  설치

설치 과정은 모델 변환을 위한 PC와 추론 실행을 위한 라즈베리파이, 두 부분으로 나뉩니다.

### **1. PC 환경 설정**

PC는 표준 PyTorch YOLOv8 모델을 NCNN 형식으로 변환하는 데 사용됩니다.

* **필수 파이썬 라이브러리 설치**:
    ```bash
    pip install ultralytics onnx onnx-simplifier
    ```

* **NCNN 변환 도구 빌드**:
    ```bash
    git clone https://github.com/Tencent/ncnn.git
    cd ncnn
    mkdir -p build && cd build
    cmake ..
    make -j$(nproc)
    ```

### **2. 라즈베리파이 환경 설정**

라즈베리파이는 추론이 실행될 대상 장치입니다.

* **기본 도구 설치**:
    ```bash
    sudo apt update && sudo apt upgrade
    sudo apt install build-essential cmake git
    ```

* **OpenCV 설치**:
    ```bash
    sudo apt install libopencv-dev
    ```

* **NCNN 라이브러리 빌드 및 설치**:
    ```bash
    git clone https://github.com/Tencent/ncnn.git
    cd ncnn
    mkdir -p build && cd build
    cmake -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=ON ..
    make -j$(nproc)
    sudo make install
    ```

---
## 실행

실행 또한 PC에서의 모델 변환과 라즈베리파이에서의 코드 컴파일 및 실행, 두 부분으로 나뉩니다.

### **1. 모델 변환 (PC에서 수행)**

NCNN 모델 파일을 얻기 위해 다음 단계를 따르세요.

* **1단계: `.pt`를 `.onnx`로 변환**:
    `export_onnx.py`와 같은 파이썬 스크립트를 아래 내용으로 생성하고 실행합니다.
    ```python
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    model.export(format='onnx', opset=10, simplify=True)
    ```

* **2단계: `.onnx`를 NCNN 형식(`.param`, `.bin`)으로 변환**:
    `ncnn/build/tools/onnx/` 디렉토리에서 빌드된 `onnx2ncnn` 도구를 사용합니다.
    ```bash
    ./tools/onnx/onnx2ncnn ../../yolov8n.onnx yolov8n.param yolov8n.bin
    ```

* **3단계: 라즈베리파이로 파일 전송**:
    생성된 `yolov8n.param`과 `yolov8n.bin` 파일을 `scp`나 다른 방법을 사용하여 라즈베리파이로 복사합니다.

### **2. 컴파일 및 실행 (라즈베리파이에서 수행)**

이제 라즈베리파이에서 C++ 추론 애플리케이션을 컴파일하고 실행합니다.

* **1단계: C++ 소스 코드(`main.cpp`) 및 `CMakeLists.txt` 준비**:
    C++ 코드(`main.cpp`)를 프로젝트 디렉토리에 배치합니다. 같은 디렉토리에 아래 내용으로 `CMakeLists.txt` 파일을 생성합니다.
    ```cmake
    cmake_minimum_required(VERSION 3.10)
    project(YoloApp)
    
    find_package(OpenCV REQUIRED)
    find_package(NCNN REQUIRED)
    
    add_executable(yolo_app main.cpp)
    target_link_libraries(yolo_app ${OpenCV_LIBS} ncnn)
    ```

* **2단계: 코드 컴파일**:
    ```bash
    mkdir build && cd build
    cmake ..
    make
    ```

* **3단계: 추론 실행 파일 실행**:
    모델 파일(`.param`, `.bin`)과 테스트 이미지가 `build` 디렉토리에 있는지 확인하고 실행합니다.
    ```bash
    ./yolo_app ../test.jpg
    ```

---
## 결과

실행 파일을 실행하면 터미널에 다음과 같은 출력이 나타나며, 이는 모델이 성공적으로 로드되고 추론이 완료되었음을 나타냅니다.

* **예상 터미널 출력**:
    ```
    NCNN 모델 로드 성공!
    추론 결과 텐서 크기: w=8400, h=84, c=1
    추론 성공! 이 결과 텐서를 해석(후처리)하여 객체를 얻을 수 있습니다.
    ```
이 출력은 프로그램이 올바르게 작동하고 있음을 확인시켜 줍니다. 다음 단계는 출력 텐서를 파싱하여 탐지된 객체의 바운딩 박스 정보를 얻는 것입니다.
