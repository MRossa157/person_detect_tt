import logging

import cv2
from tqdm import tqdm
from ultralytics import YOLO


class PeopleTracker():
    def __init__(self, model_path, ) -> None:
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        logging.info('Поиск объектов на кадрах')
        people_detections = []

        for frame in tqdm(frames, desc='Ищем людей на видео'):
            people_dict = self.__detect_frame(frame)
            people_detections.append(people_dict)

        return people_detections

    def draw_bboxes(self, video_frames, player_detections):
        logging.info('Отрисовываем границы людей на видео')
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Отрисовка Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"People {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames

    def __detect_frame(self, frame):
        # persist = True - означает, что данный кадр не единичный, а взят из видео и мы подадим следующий кадр
        results = self.model.track(frame, persist=True, verbose=False)[0]
        id_name_dict = results.names

        people_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]

            if object_cls_name == 'person':
                people_dict[track_id] = result

        return people_dict