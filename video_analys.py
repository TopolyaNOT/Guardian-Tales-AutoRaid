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


        y_start, y_end = self.roi[0][0], self.roi[0][1]
        x_start, x_end = self.roi[1][0], self.roi[1][1]
    
        roi = frame[y_start:y_end, x_start:x_end]

        if self.to_gray:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(roi, self.rgb_template, cv2.TM_CCOEFF_NORMED, mask=self.mask)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= self.threshold:
            h, w = self.rgb_template.shape[:2]
            center_x = max_loc[0] + x_start + int(w//2)
            center_y = max_loc[1] + y_start + int(h//2)

            top_left = (max_loc[0] + x_start, max_loc[1] + y_start)
            bottom_right = (top_left[0] + w, top_left[1] + h)
            
            
            cv2.rectangle(img=frame, pt1=top_left, pt2=bottom_right, color=(0, 255, 0), thickness=2)

            # Для візуального тестування
            # cv2.circle(img=frame, center=(center_x, center_y), color=(0,255,0), radius=10, thickness=2)
            # cv2.imshow("Result", frame)
            # cv2.waitKey()

            return center_x, center_y
        else:
            print("❌ Skill not found")
        



if __name__ == "__main__":
    
    main_skill = PatternAnalys(
        template = "assets/main_skill.png",
        roi = ((500,850), (1350,1700)),
        to_gray=True,
        threshold=0.2)

    frame = cv2.imread("assets/test.png")
    if frame is not None:
        print(main_skill.start(frame=frame))