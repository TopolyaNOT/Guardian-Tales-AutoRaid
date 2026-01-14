import cv2
import numpy as np
from typing import Tuple, Optional, List, Any



ROI_THICKNESS = 1
DETECTION_THICKNESS = 2
TOUCH_RADIUS = 5

class Colors:
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)

class PatternAnalyzer:
    def __init__(self,
                name: str,
                template: str,
                roi: Tuple[Tuple[int, int], Tuple[int,int]],
                to_gray: bool = True,
                threshold: float = 0.2):
        
        """
        Аналізатор для пошуку шаблонів на зображеннях методом template matching.
        
        Args:
            template:     Шлях до PNG-файлу з альфа-каналом
            roi:          Область інтересу ((y_start, y_end), (x_start, x_end))
            to_gray:      Конвертувати у відтінки сірого
            name:         Ім'я детектора для логування
            threshold:    Поріг схожості (0.0-1.0)
        """

        self.name = name
        self.roi = roi
        self.threshold = threshold

        template_raw = cv2.imread(template, cv2.IMREAD_UNCHANGED)
        if template_raw is None:
            print("Помилка завантаження шаблону!")
            raise FileNotFoundError(f"Не вдалося завантажити шаблон: {template}")

        self.bgr_template = template_raw[:, :, 0:3]
        self.mask = template_raw[:, :, 3]
        self.h, self.w = self.bgr_template.shape[:2]

        if to_gray:
            self.bgr_template = cv2.cvtColor(self.bgr_template, cv2.COLOR_BGR2GRAY)
        
        self.to_gray = to_gray



    def find(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]]:

        """
        Функція для пошуку шаблонів на зображеннях методом template matching.
        

        :param frame:  Кадер з відео
        :type frame:   np.ndarray
        :return:       (
                        (центер шаблону по X, центер шаблону по Y), 
                        (верхній лівий край (x ,y), правий нижній край (x, y))
                        )

        """


        y_start, y_end = self.roi[0]
        x_start, x_end = self.roi[1]
    
        roi = frame[y_start:y_end, x_start:x_end]

        if self.to_gray:
            
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        

        res = cv2.matchTemplate(roi, self.bgr_template, cv2.TM_CCOEFF_NORMED, mask=self.mask)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= self.threshold:
            h, w = self.bgr_template.shape[:2]
            center_x = max_loc[0] + x_start + int(w//2)
            center_y = max_loc[1] + y_start + int(h//2)

            top_left = (max_loc[0] + x_start, max_loc[1] + y_start)
            bottom_right = (top_left[0] + int(w), top_left[1] + int(h))

            return ((center_x, center_y), (top_left, bottom_right))
        return None
    

    def draw_touch(self, frame: np.ndarray, coords: Tuple[int, int]) -> None:
        """        
        Малює на кадрі коло - зону дотику

        Args:
            frame: Кадр для малювання (змінюється in-place)
        """

        if frame is None:
            return
        if coords is None:
            return

        cv2.circle(img=frame, center=coords, color=Colors.GREEN, radius=TOUCH_RADIUS, thickness=DETECTION_THICKNESS)


    def draw_roi(self, frame: np.ndarray) -> None:
        """
        Малює на кадрі зону roi
        
        Args:
            frame: Кадр для малювання (змінюється in-place)
        """

        y_start, y_end = self.roi[0]
        x_start, x_end = self.roi[1]

        cv2.rectangle(img=frame, pt1=(x_start, y_start), pt2=(x_end, y_end), color=Colors.RED, thickness=ROI_THICKNESS)
        cv2.putText(img=frame, text=self.name, org=(x_start, y_start - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=Colors.RED, thickness=DETECTION_THICKNESS)


    def draw_detection(self, frame: np.ndarray, pt: Tuple[Tuple[int, int], Tuple[int, int]]) -> None:
        """
        Малює на кадрі зону відповідну шаблону

        Args:
            frame: Кадр для малювання (змінюється in-place)
        """
        
        if frame is None:
            return
        if pt is None:
            return 
        cv2.rectangle(img=frame, pt1=pt[0], pt2=pt[1], color=Colors.GREEN, thickness=DETECTION_THICKNESS)
        cv2.putText(img=frame, text=self.name, org=(pt[0][0], pt[0][1] - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=Colors.GREEN, thickness=1)


class VideoAnalyzer:
    def __init__(self, video_path: str):
        self.video = cv2.VideoCapture(video_path)

    def run(self, detectors: List[PatternAnalyzer]) -> None:
        """
        Запуск аналізатора:

        detectors: Список детекторів які необхідно шукати
        type detectors: List[PatternAnalyzer]
        """
        try:
            while self.video.isOpened():
                success, frame = self.video.read()

                if not success:
                    break

                for detector in detectors:
                    coords = detector.find(frame)
                    if coords:
                        detector.draw_detection(frame=frame, pt=coords[1])
                        # print(f"[{detector.name}] знайдено в {coords[0]}")


                # Малювання зон у відео
                cv2.imshow("Detector analyzer", frame)

                # Кнопка виходу
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Помилка під час запуску аналізу відео: {e}")
            









if __name__ == "__main__":


    
    main_skill = PatternAnalyzer(
        name="Main skill",
        template = "assets/main_skill.png",
        roi = ((575,800), (1400, 1650)),
        to_gray=True,
        threshold=0.1)
    
    main_attack = PatternAnalyzer(
        name="Main attack",
        template="assets/main_attack.png",
        roi=((800, 990), (1350, 1550)),
        to_gray=True,
        threshold=0.3
    )

    second_attack = PatternAnalyzer(
        name="Second attack",
        template="assets/second_attack.png",
        roi=((500, 700), (1600, 1800)),
        to_gray=True,
        threshold=0.1
    )

    orca_skill = PatternAnalyzer(
        name="Hero skill",
        template="assets/orca_skill.png",
        roi=((300, 500), (1625, 1825)),
        to_gray=True,
        threshold=0.1
    )



    video_analys = VideoAnalyzer("assets/video2.mkv")
    video_analys.run([orca_skill, main_attack, main_skill, second_attack])