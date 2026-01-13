import cv2
import numpy as np
from typing import Tuple, Optional



class PatternAnalys:
    def __init__(self,
                template:str,
                roi:Tuple[Tuple[int, int], Tuple[int,int]],
                to_gray:bool = True,
                threshold: float = 0.2):
        
        self.roi = roi
        self.threshold = threshold

        template_raw = cv2.imread(template, cv2.IMREAD_UNCHANGED)
        if template_raw is None:
            print("Помилка завантаження шаблону!")
            exit()

        self.rgb_template = template_raw[:, :, 0:3]
        self.mask = template_raw[:, :, 3]
        self.h, self.w = self.rgb_template.shape[:2]

        if to_gray:
            self.rgb_template = cv2.cvtColor(self.rgb_template, cv2.COLOR_BGR2GRAY)
        
        self.to_gray = to_gray



    def start(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:

        y_start, y_end = self.roi[0]
        x_start, x_end = self.roi[1]
    
        roi = frame[y_start:y_end, x_start:x_end]

        if self.to_gray:
            
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        

        res = cv2.matchTemplate(roi, self.rgb_template, cv2.TM_CCOEFF_NORMED, mask=self.mask)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= self.threshold and max_val != 0 and max_loc != []:
            h, w = self.rgb_template.shape[:2]
            center_x = max_loc[0] + x_start + int(w//2)
            center_y = max_loc[1] + y_start + int(h//2)

            top_left = (max_loc[0] + x_start, max_loc[1] + y_start)
            bottom_right = (top_left[0] + w, top_left[1] + h)
            
            
            cv2.rectangle(img=frame, pt1=top_left, pt2=bottom_right, color=(0, 255, 0), thickness=2)                    # Зона кнопки
            # print(f"🔍 схожість {max_val * 100:.2f}%")
            # Для візуального тестування
            cv2.rectangle(img=frame, pt1=(x_start, y_start), pt2=(x_end, y_end), color=(0, 0, 255), thickness=2)        # Зона Roi
            cv2.circle(img=frame, center=(center_x, center_y), color=(0,255,0), radius=5, thickness=2)                  # Точка натиску
            cv2.imshow("Result", frame)
            cv2.waitKey()


            return center_x, center_y
        else:
            print("❌ Skill not found")

class VideoAnalys:
    def __init__(self, video_path:str):
        self.video = cv2.VideoCapture(video_path)

    def start_analys(self, action: PatternAnalys, on_frame: int = 5) -> int:

        frame_count = 0
        button_count = 0

        while self.video.isOpened():
            success, frame = self.video.read()

            if not success:
                break

            if frame_count % on_frame == 0:
                cord = action.start(frame)
                if cord is not None:
                    button_count += 1


            frame_count += 1
        return button_count









if __name__ == "__main__":

    
    main_skill = PatternAnalys(
        template = "assets/main_skill.png",
        roi = ((575,800), (1400, 1650)),
        to_gray=True,
        threshold=0.1)
    
    main_attack = PatternAnalys(
        template="assets/main_attack.png",
        roi=((800, 990), (1350, 1550)),
        to_gray=True,
        threshold=0.3
    )

    second_attack = PatternAnalys(
        template="assets/second_attack.png",
        roi=((500, 700), (1600, 1800)),
        to_gray=True,
        threshold=0.1
    )

    orca_skill = PatternAnalys(
        template="assets/orca_skill.png",
        roi=((300, 500), (1625, 1825)),
        to_gray=True,
        threshold=0.1
    )



    video_analys = VideoAnalys("assets/video2.mkv")
    button_count = video_analys.start_analys(orca_skill, 100)

    print(button_count)
