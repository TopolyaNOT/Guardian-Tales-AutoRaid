import cv2
import numpy as np


# path = r"C:\Users\Afina\Downloads\Мінотав вода.mp4"

# video = cv2.VideoCapture(path)
# fps = video.get(cv2.CAP_PROP_FPS)

# frame_count = 0

# while video.isOpened():
#     success, frame = video.read()
#     if not success:
#         break
#     if frame_count % 5 == 0:
#         pass
#     frame_count += 1

# print(frame_count)


template = cv2.imread("assets\\main_skill.png")
frame = cv2.imread("assets\\test.png")

res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

threshold = 0.8
loc = np.where(res >= threshold)

print(loc)
if len(loc[0]) > 0:
    print("Skill active")