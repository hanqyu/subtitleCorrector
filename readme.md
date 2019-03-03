# subtitleCorrector

(아직 진행 중인 프로젝트입니다)

### Introduction
영상 스트리밍 서비스에서 __Closed Caption 자막__(켜고 끌 수 있는 자막)을 다룰때 생길 수 있는 문제가 여럿 있음

1. 영상 위에 이미 입혀진 텍스트와 겹쳐서 자막이 안 보이는 경우
    ex) open caption 혹은 오프닝 크레딧
2. 자막의 싱크가 맞질 않는 경우
3. 극 중 인물 간의 존칭 관계가 맞지 않거나 매번 달라지는 경우.
    ex) A와 B가 동등한 관계임에도 특정 인물만 존댓말을 사용. 혹은 에피소드마다 존댓말 관계가 바뀜

자막을 하나하나 검수하는 것이 인적 리소스가 많이 드는 일이므로 이를 해결하기 위한 솔루션을 마련함

opencv와 tensorflow를 기반으로 히여, Deep learning을 활용한 Computer vision 기술이 될 것

<br><br>
## 1. 영상 내 텍스트 검출
### main.py

    python Users/username/subtitleCorrector/main.py 
    --video_dir Users/username/subtitleCorrector/video/ \ 
    --subtitle_dir Users/username/subtitleCorrector/subtitle/
    
* 자막 파일 import 후, 자막 구간을 추출해서 각 초마다의 프레임을 추출함.
    * 자막 파일의 포맷은 Webvtt 포맷(.vtt)을 기준으로 작성함
* 자막 글자수를 기준으로 자막이 표시될 것으로 예상되는 영역을 프레임 이미지에서 crop함
    * 모바일 화면을 기준으로 했으므로 조금 더 넓은 영역을 판단하게 됨
* 추출한 영역에 텍스트가 존재하는지 여부는 tensorflow Imagenet을 활용. 속도와 효율성을 위해 MobilNet Architecture를 사용

todo: 텍스트가 있다고 판단되는 경우, 위치를 옮길 곳의 텍스트를 검출해내는 과정 추가 필요
todo: 검수가 완료된 자막을 쉽고 빠르게 검수할 수 있는 기능 

##### result)
* 이미지가 있다고 판단되는 자막의 구간에는 position:10%을 넣어주게 됨
* 자막이 표시되는 영역을 상단으로 위치를 vtt포맷에 맞게 바꾸는 것
<br><br>
* 모델을 돌린 프레임 이미지를 따로 저장하여 나중에 재학습할 수 있도록 함

<br>

### retrain.py
: Tensorflow MobileNet 모델 training

__/train_images/__ 의 하위 디렉토리에는 __text__와 __non text__ 두 개의 폴더에 학습할 이미지들을 담고 있어야 함

    python Users/username/subtitleCorrector/retrain.py \
    --image_dir Users/username/subtitleCorrector/train_images/ \
    --how_many_training_steps=1000 \
    --architecture mobilenet_1.0_224

<br><br>
## Reference
### [nicewoong / pyTextGotcha](https://github.com/nicewoong/pyTextGotcha)
* Canny Edge Dectecting을 위한 과정에서 많은 부분 참조, 사용했습니다.

### [Naver D2 blog post](https://d2.naver.com/helloworld/8344782)
* 딥러닝과 OpenCV를 활용해 사진 속 글자 검출하기

### [Tensorflow/ImageNet(InceptionV3)](https://github.com/tensorflow/tensorflow/blob/c565660e008cf666c582668cb0d0937ca86e71fb/tensorflow/examples/image_retraining/retrain.py)
* Tensorflow의 Inception V3 모델을 retrain 하는 예제.
* 이 모델의 사용법은 [솔라리스의 인공지능 연구실](http://solarisailab.com/archives/1422)을 참조함

### [Sumit Kumar Arora의 Medium Article](https://medium.com/@sumit.arora/training-a-neural-network-using-mobilenets-in-tensorflow-for-image-classification-on-android-14f2792f64c1)
* Training a neural network using Mobilenets in TensorFlow for image classification on Android
* 더 나은 학습속도와 더 빠른 성능의 모델을 위헤 MobileNet을 사용함.