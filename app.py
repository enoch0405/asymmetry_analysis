import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Streamlit 타이틀 설정
st.markdown(
    """
    <h1 style='text-align: center; color: #6A5ACD; font-family: Arial, sans-serif;'>
        Head and Shoulder Asymmetry
    </h1>
    """,
    unsafe_allow_html=True
)

# Mediapipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def initialize_camera():
    """웹캠을 초기화합니다."""
    return cv2.VideoCapture(0)

def calculate_distance(point1, point2):
    """두 점 사이의 유클리드 거리를 계산합니다."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_slope(point1, point2):
    """두 점 사이의 기울기를 계산합니다."""
    if point2[0] - point1[0] == 0:
        return 0  # 수직일 때 기울기 무한대 방지
    return (point2[1] - point1[1]) / (point2[0] - point1[0])

def process_frame(frame, pose):
    """프레임을 처리하고 포즈 분석 결과를 반환합니다."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    return results

def draw_landmarks(frame, landmarks):
    """랜드마크를 프레임에 그립니다."""
    # 전체 랜드마크 그리기 (초록색으로 어깨선 및 다른 신체 부분 그리기)
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # 초록색
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # 초록색
    )

    # 얼굴 부분만 파란색으로 그리기
    face_landmarks = [
        mp_pose.PoseLandmark.NOSE, 
        mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.LEFT_EYE,
        mp_pose.PoseLandmark.LEFT_EYE_OUTER, mp_pose.PoseLandmark.RIGHT_EYE_INNER,
        mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
        mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.MOUTH_LEFT, mp_pose.PoseLandmark.MOUTH_RIGHT
    ]

    for landmark in face_landmarks:
        idx = landmark.value
        point = (int(landmarks.landmark[idx].x * frame.shape[1]), int(landmarks.landmark[idx].y * frame.shape[0]))
        cv2.circle(frame, point, 3, (255, 0, 0), -1)  # 파란색으로 얼굴 랜드마크 점

    # 얼굴의 랜드마크 연결 선을 파란색으로 그리기
    face_connections = [
        (mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.RIGHT_EYE_INNER),
        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE_INNER),
        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_EYE_INNER),
        (mp_pose.PoseLandmark.LEFT_EYE_OUTER, mp_pose.PoseLandmark.LEFT_EAR),
        (mp_pose.PoseLandmark.RIGHT_EYE_OUTER, mp_pose.PoseLandmark.RIGHT_EAR),
        (mp_pose.PoseLandmark.MOUTH_LEFT, mp_pose.PoseLandmark.MOUTH_RIGHT)
    ]

    for connection in face_connections:
        start_idx = connection[0].value
        end_idx = connection[1].value
        start_point = (int(landmarks.landmark[start_idx].x * frame.shape[1]), int(landmarks.landmark[start_idx].y * frame.shape[0]))
        end_point = (int(landmarks.landmark[end_idx].x * frame.shape[1]), int(landmarks.landmark[end_idx].y * frame.shape[0]))
        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)  # 파란색 선

def calculate_asymmetry(landmarks):
    """비대칭성을 계산하고 결과를 반환합니다."""
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
    
    # 어깨 중심 좌표 계산
    shoulder_center = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]

    # 어깨 비대칭 계산 (y 좌표 차이)
    shoulder_asymmetry = left_shoulder[1] - right_shoulder[1]

    # 머리와 어깨 중심의 비대칭 계산 (x 좌표 차이)
    head_to_shoulder_center = nose[0] - shoulder_center[0]

    return shoulder_asymmetry, head_to_shoulder_center

def display_results(frame, shoulder_asymmetry, head_to_shoulder_center):
    """비대칭 결과를 프레임에 출력합니다."""
    cv2.putText(frame, f'Shoulder Asymmetry: {shoulder_asymmetry:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Head to Shoulder Center: {head_to_shoulder_center:.2f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

def main():
    """메인 함수: 웹캠을 사용하여 포즈를 분석하고 결과를 출력합니다."""
    stframe = st.empty()
    cap = initialize_camera()
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 처리
            results = process_frame(frame, pose)

            # 포즈 랜드마크가 검출되었을 때
            if results.pose_landmarks:
                draw_landmarks(frame, results.pose_landmarks)

                # 비대칭성 계산
                shoulder_asymmetry, head_to_shoulder_center = calculate_asymmetry(results.pose_landmarks.landmark)

                # 결과 출력
                display_results(frame, shoulder_asymmetry, head_to_shoulder_center)

            # 프레임을 BGR에서 RGB로 변환 (스트림릿에서 사용하기 위해)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Streamlit에 비디오 프레임 표시
            stframe.image(frame, channels='RGB', use_column_width=True)

    cap.release()

if st.button("START/STOP"):
    main()
