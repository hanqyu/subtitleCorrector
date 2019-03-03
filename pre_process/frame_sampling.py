'''
영상에서 무작위 50개의 프레임 이미지를 추출한다
'''

import os
import cv2
import random


if __name__ == "__main__":
    video_samples = ['sample/sample1.mp4', 'sample/sample2.mp4', 'sample/sample3.mp4']
    count = 0
    for video_sample in video_samples:
        cap = cv2.VideoCapture(video_sample)
        frame_samples = sorted(random.sample(list(range(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))), 50))

        success, image = cap.read()

        while success:
            for frame in frame_samples:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                success, image = cap.read()
                # print('Read a new frame: ', success)
                file_name = os.path.join('sample/sample/', 'frame%d.jpg' % count)
                save = cv2.imwrite(file_name, image)
                if save is True:
                    print('%s has been saved.' % file_name)  # save frame as JPEG file
                count += 1
            break

