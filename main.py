import logging

from trackers import PeopleTracker
from utils import read_video, save_video


def main():
    logging.basicConfig(level=logging.INFO)

    # Считываем видео
    logging.info('Начали считывать видео')
    input_video_path = "input_videos/crowd.mp4"
    video_frames = read_video(input_video_path)

    # Вычисляем людей
    people_tracker = PeopleTracker(model_path='yolov8x')
    people_detections = people_tracker.detect_frames(video_frames)

    # Отрисовываем границы людей
    output_video_frames = people_tracker.draw_bboxes(video_frames, people_detections)

    save_video(output_video_frames, "output_videos/result.avi")

if __name__ == "__main__":
    main()