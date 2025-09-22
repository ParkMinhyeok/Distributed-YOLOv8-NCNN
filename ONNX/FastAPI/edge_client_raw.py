# rpi_client.py
import cv2
import requests
import time

# PC 서버의 IP 주소와 포트를 설정하세요.
SERVER_URL = "http://192.168.0.36:8000"

def main():
    # OpenCV를 사용하여 웹캠(카메라 인덱스 0)을 엽니다.
    # Raspberry Pi에 따라 카메라 인덱스가 다를 수 있습니다.
    cap = cv2.VideoCapture(1)
    
    # 웹캠이 제대로 열렸는지 확인
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam.")
                break

            # 프레임을 JPEG로 인코딩
            _, jpeg_data = cv2.imencode('.jpg', frame)
            
            # 서버로 이미지 데이터 전송
            try:
                response = requests.post(f"{SERVER_URL}/video", data=jpeg_data.tobytes(), 
                                         headers={'Content-Type': 'image/jpeg'}, timeout=5)
                if response.status_code == 200:
                    print("Image frame sent successfully!")
                else:
                    print(f"Server error: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Error communicating with server: {e}")

            # 키 입력 대기 (선택 사항: 'q'를 누르면 종료)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 웹캠 닫기
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
