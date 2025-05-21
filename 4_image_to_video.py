import cv2
import numpy as np
import os
'''
def create_moving_crop_video(image_path, crop_filename="crop/crop_video.mp4", crop_size=(100, 100), speed=2, fps=30):
    # 이미지 불러오기
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    
    h, w, _ = img.shape
    crop_w, crop_h = crop_size

    # 시작 위치: 왼쪽 아래
    x, y = 0, h - crop_h

    # 끝 위치: 오른쪽 위
    end_x, end_y = w - crop_w, 0

    # 이동 방향 벡터 정규화
    dx = end_x - x
    dy = end_y - y
    dist = (dx**2 + dy**2)**0.5
    dir_x = dx / dist
    dir_y = dy / dist

    # 프레임 수 계산
    steps = int(dist // speed)

    # 저장 폴더가 없으면 생성
    os.makedirs(os.path.dirname(crop_filename), exist_ok=True)

    # 비디오 라이터 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(crop_filename, fourcc, fps, crop_size)

    for i in range(steps):
        cx = int(x + dir_x * speed * i)
        cy = int(y + dir_y * speed * i)

        if cx + crop_w > w or cy + crop_h > h or cx < 0 or cy < 0:
            break

        crop = img[cy:cy + crop_h, cx:cx + crop_w]
        out.write(crop)

    out.release()
    print(f"영상이 '{crop_filename}'에 저장되었습니다.")


create_moving_crop_video("mountain_image.jpg", "crop/crop_video.mp4", crop_size=(400, 400), speed=5, fps=30)
'''
'''
import cv2
import numpy as np
import os

def create_moving_crop_video(image_path, crop_filename="crop/crop_video.mp4", crop_size=(100, 100), acceleration=1, fps=30):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    
    h, w, _ = img.shape
    crop_w, crop_h = crop_size

    # 시작 위치: 왼쪽 아래
    x0, y0 = 0, h - crop_h
    x1, y1 = w - crop_w, 0

    dx = x1 - x0
    dy = y1 - y0
    total_dist = (dx**2 + dy**2)**0.5
    dir_x = dx / total_dist
    dir_y = dy / total_dist

    # 누적 거리 s = 0.5 * a * t^2 => 최대 프레임 수 t_max = sqrt(2 * s / a)
    max_frames = int((2 * total_dist / acceleration) ** 0.5)

    os.makedirs(os.path.dirname(crop_filename), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(crop_filename, fourcc, fps, crop_size)

    for i in range(max_frames):
        # 이동 거리 s = 0.5 * a * i^2
        dist = 0.5 * acceleration * (i ** 2)
        cx = int(x0 + dir_x * dist)
        cy = int(y0 + dir_y * dist)

        if cx + crop_w > w or cy + crop_h > h or cx < 0 or cy < 0:
            break

        crop = img[cy:cy + crop_h, cx:cx + crop_w]
        out.write(crop)

    out.release()
    print(f"가속도 {acceleration}로 생성된 영상이 '{crop_filename}'에 저장되었습니다.")

create_moving_crop_video("mountain_image.jpg", "crop/crop_acc.mp4", crop_size=(400, 400), acceleration=1, fps=30)
'''
import cv2
import numpy as np
import os

def create_moving_crop_video(image_path, crop_filename="crop/crop_acc_change.mp4",
                              crop_size=(100, 100), fps=30, switch_interval=30):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    h, w, _ = img.shape
    crop_w, crop_h = crop_size

    # 시작 위치: 왼쪽 아래
    x0, y0 = 0, h - crop_h
    x1, y1 = w - crop_w, 0

    dx = x1 - x0
    dy = y1 - y0
    total_dist = (dx**2 + dy**2)**0.5
    dir_x = dx / total_dist
    dir_y = dy / total_dist

    os.makedirs(os.path.dirname(crop_filename), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(crop_filename, fourcc, fps, crop_size)

    # 속도 조절용 변수
    speed_base = 5.0
    speed_variation = 5.0
    period = switch_interval

    cumulative_dist = 0
    frame_idx = 0

    while True:
        # 속도 진동 계산 (한 주기마다 속도 오르내림)
        t = frame_idx / period
        speed = speed_base + speed_variation * np.sin(2 * np.pi * t)

        cumulative_dist += speed
        cx = int(x0 + dir_x * cumulative_dist)
        cy = int(y0 + dir_y * cumulative_dist)

        if cx + crop_w > w or cy + crop_h > h or cx < 0 or cy < 0:
            break

        crop = img[cy:cy + crop_h, cx:cx + crop_w]
        out.write(crop)
        frame_idx += 1

    out.release()
    print(f"속도가 주기적으로 변하는 영상이 '{crop_filename}'에 저장되었습니다.")

create_moving_crop_video(
    "mountain_image.jpg",
    "crop/crop_acc_change.mp4",
    crop_size=(400, 400),
    fps=30,
    switch_interval=20
)