# -*- coding: utf-8 -*-
'''
모델 훈련을 위해 뽑아낸 프레임(frame_sampling)에서
text로 의심되는 contour들을 뽑아내고 저장한다
'''
import process_image
import glob
import cv2
import os

if __name__ == "__main__":
    image_path = 'sample/sample/*.jpg'
    save_path = 'sample/train/'

    samples = glob.glob(image_path)
    count = 0
    for frame_path in samples:
        frame = process_image.open_image(frame_path)
        contours = process_image.process_image(frame)
        results = process_image.get_cropped_images(frame, contours)

        for result in results:
            morph = result.copy()
            morph = process_image.get_gray(morph)
            morph = process_image.get_canny(morph)
            morph = process_image.get_gradient(morph)

            file_name = os.path.join(save_path, 'train_%d.jpg' % count)
            save = cv2.imwrite(file_name, morph)
            if save is True:
                print('%s has been saved.' % file_name)  # save frame as JPEG file
                count += 1
