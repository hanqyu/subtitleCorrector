# -*- coding: utf-8 -*-

import process_image, process_subtitle, label_image
import cv2
import os
import numpy as np
import tensorflow as tf
from datetime import datetime

modelFullPath = '/tmp/output_graph.pb'         # 읽어들일 graph 파일 경로
labelsFullPath = '/tmp/output_labels.txt'      # 읽어들일 labels 파일 경로


def create_graph():
    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def cv2_image_to_tensorflow(cv2_image):
    image_data = cv2.imencode('.jpg', cv2_image)[1].tostring()
    return image_data


def parse_subtitle_file(subtitle_path):
    with open(subtitle_path, 'r') as f:
        data = f.read()
        _sub = data
    _sub = _sub.split('\n\n')
    result = list()
    for x in _sub:
        parse = list(filter(None, x.split('\n')))
        if len(parse) > 1:
            start, end = parse[0].split(' --> ')
            _dict = {
                    'time': parse[0],
                    'start': total_seconds(start),
                    'end': total_seconds(end),
                    'text': '  \n'.join(parse[1:]),
                    'video_on_text': False
            }
            if 'position:10%' in _dict['time']:
                _dict['video_on_text'] = True
            result.append(_dict)

    return result


def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap


def parse_timestamp(time):
    hour = 0
    parsed = time.split(':')
    if len(parsed) == 3:
        hour = parsed[0]
        minute = parsed[1]
    else:
        minute = parsed[0]
    second = parsed[-1].split('.')[0]

    return int(hour), int(minute), int(second)


def total_seconds(time):
    hour, minute, second = parse_timestamp(time)
    return (hour * 60 * 60) + (minute * 60) + second


def unparse_subtitle_file(list_subtitle):
    result = 'WEBVTT\n\n'
    for line in list_subtitle:
        if line['video_on_text']:
            result += line['time'] + ' position:10%' + '\n' + line['text'] + '\n\n'
        else:
            result += line['time'] + '\n' + line['text'] + '\n\n'
    return result


def save_subtitle_file(string, file_path, file_name):
    path = os.path.join(file_path, file_name)
    with open(path, 'w') as f:
        f.write(string)


def run_inference_on_image(cv2_image):
    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        image_data = cv2_image_to_tensorflow(cv2_image)

        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})

        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "").replace("b'","").replace("\\n'","") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))

        answer = labels[top_k[0]]
        return answer


if __name__ == "__main__":
    file_path = 'sample/'
    file_name = 'sample1.vtt'

    model_file = "/tmp/output_graph.pb"
    label_file = "/tmp/output_labels.txt"

    subtitle = parse_subtitle_file(os.path.join(file_path, file_name))
    cap = open_video('sample/sample1.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)

    success, first_image = cap.read()

    working_time = datetime.now()
    print(working_time)

    graph = label_image.load_graph(modelFullPath)

    create_graph()

    with tf.Session() as sess:
        for i in range(0, len(subtitle)):
            line = subtitle[i]
            print('작업: %s' % line['time'])
            frames = [int(x * fps) for x in list(range(line['start'], line['end']))]
            frames = list(set(frames))
            count = 0
            for frame in frames:
                video_on_text = subtitle[i]['video_on_text']
                if video_on_text:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                success, image = cap.read()

                if success:
                    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

                    subtitle_region = process_subtitle.get_subtitle_region(image, line['text'])
                    image_crop = process_image.crop_image(image, subtitle_region)
                    processed_image = process_image.process_image(image_crop, no_contours=True)

                    image_data = cv2_image_to_tensorflow(processed_image)
                    results = sess.run(softmax_tensor,
                                       {'DecodeJpeg/contents:0': image_data})
                    results = np.squeeze(results)

                    top_k = results.argsort()[-5:][::-1]
                    labels = label_image.load_labels(label_file)
                    answer = labels[top_k[0]]

                    if answer == 'text':
                        subtitle[i]['video_on_text'] = True
                        print('find!')
                        cv2.imwrite('result/text/%s_%s_%d.jpg' % (file_name, line['time'], count), processed_image)
                        count += 1
                        break
                    elif answer == 'non text':
                        cv2.imwrite('result/non text/%s_%s_%d.jpg' % (file_name, line['time'], count), processed_image)
                        count += 1
                    else:
                        print('WARNING: cannot find image prediction')

    if subtitle != parse_subtitle_file('sample/sample1.vtt'):
        print('something changed in subtitle!')
        save_subtitle_file(unparse_subtitle_file(subtitle), 'result/', file_name)
        print('successfully saved')
    else:
        print('nothing found on video image')
    working_time = datetime.now() - working_time
    print('작업 종료: %s분' % int(working_time.total_seconds()/60))

