
_subtitle = {
    'text_size': [50/1334, 55/750],
    'line_margin': 10/750,
    'padding': [10/1334, 10/750],
    'location': {'1_line': 650/750, '2_line': 585/750}
}


def resize_subtitle(subtitle, width, height):
    _dict = dict()
    _dict['text_size'] = int(subtitle['text_size'][0] * width), int(subtitle['text_size'][1] * height)
    _dict['line_margin'] = int(subtitle['line_margin'] * height)
    _dict['padding'] = int(subtitle['padding'][0] * width), int(subtitle['padding'][1] * height)
    _dict['location'] = {
        '1_line': int(subtitle['location']['1_line'] * height),
        '2_line': int(subtitle['location']['2_line'] * height)
    }

    return _dict


def is_not_resoultion_16_9(image_origin):
    height, width = image_origin.shape[:2]

    # 16:9 화면 비율보다 5% 이상 더 큰지 아닌지
    boolean = width / height - 1.777 > 1.777 * 0.05
    return boolean


def difference_height_to_16_9(image_origin):
    height, width = image_origin.shape[:2]
    return int(width/(1920/1080) - height)


def finalize_subtitle_region(og_height, delta, y, height):
    if delta / 2 - (og_height - height - y) > 0:
        height -= delta / 2 - (og_height - height - y)
        height = int(height)
    y -= int(delta / 2)
    return y, height


def get_subtitle_region(image_origin, text):
    global _subtitle
    image_copy = image_origin.copy()
    og_height, og_width = image_copy.shape[:2]
    if is_not_resoultion_16_9(image_copy):
        delta = difference_height_to_16_9(image_copy)
        og_height += delta

    _sub = _subtitle.copy()
    _sub = resize_subtitle(_sub, og_width, og_height)

    text_length = max([len(x.strip()) for x in text.split('\n')])

    width = text_length * _sub['text_size'][0]
    width += _sub['padding'][0] * 2

    x = (og_width - width) / 2
    x = int(x)

    height = _sub['text_size'][1]

    if len(text.split('\n')) == 2:
        height = height * 2 + _sub['line_margin']
        height += _sub['padding'][1] * 2
        y = _sub['location']['2_line'] - _sub['padding'][1]
    else:
        y = _sub['location']['1_line'] - _sub['padding'][1]

    if is_not_resoultion_16_9(image_copy):
        y, height = finalize_subtitle_region(og_height, delta, y, height)

    return x, y, width, height