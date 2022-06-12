## OCR Text Detection Competition
&nbsp;&nbsp;OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 컴퓨터가 인식할 수 있도록 하는 기술로 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 task로 이루어져 있다. 본 프로젝트는 OCR task 중 ‘글자 검출’ task만을 대회 형식으로 진행했던 프로젝트이다.      
&nbsp;&nbsp;다른 대회와는 다르게 Model 수정이 불가하고 Data 추가, Augmentation 변경만 가능했던 Data-centric Competiton이다.
- Input: 글자가 포함된 전체 이미지
- Output: bbox 좌표가 포함된 UFO Format

![image](https://user-images.githubusercontent.com/81875412/173242476-607ffe10-79d7-4858-aa31-d563581d729a.png) <br>
(출처 : 위키피디아)


## 💁TEAM
### CV 17조 MG세대
|민선아|백경륜|이도연|이효석|임동우|
| :--------: | :--------: | :--------: | :--------: | :--------: |
|<img src="https://user-images.githubusercontent.com/78402615/172766340-8439701d-e9ac-4c33-a587-9b8895c7ed07.png" width="120" height="120"/>|<img src="https://user-images.githubusercontent.com/78402615/172766371-7d6c6fa3-a7cd-4c21-92f2-12d7726cc6fc.png" width="120" height="120"/>|<img src="https://user-images.githubusercontent.com/78402615/172784450-628b30a3-567f-489a-b3da-26a7837167af.png" width="120" height="120"/>|<img src="https://user-images.githubusercontent.com/78402615/172766321-3b3a4dd4-7428-4c9f-9f91-f69a14c9f8cc.png" width="120" height="120"/>|<img src="https://user-images.githubusercontent.com/78402615/172766404-7de4a05a-d193-496f-9b6b-5e5bdd916193.png" width="120" height="120"/>|
|[@seonahmin](https://github.com/seonahmin)|[@baekkr95](https://github.com/baekkr95)|[@omocomo](https://github.com/omocomo)|[@hyoseok1223](https://github.com/hyoseok1223)|[@Dongwoo-Im](https://github.com/Dongwoo-Im)|
|Wandb Image Logging|Augmentation<br>코드 작성 및 실험|Inference 시각화<br>Dataset 실험|Datatset 시각화<br>Dataset 코드 작성 및 실험|Baseline 코드 작성<br>Wandb Metric Logging|

## 🏆Leaderboard Ranking
- Public: 4등, f1 0.7133, recall 0.6183, precision 0.8428
- Private: 6등, f1 0.6833, recall 0.5883, precision 0.8148

## 🧪Experiment
1. Dataset : ICDAR 19, AIHub Dataset 추가 확보 및 실험
2. Augmentation : Geometric, Color transfortm 및 Multi-scale Augmentation 실험

## 📖 Reference
* ICDAR dataset
    * 2017 : https://rrc.cvc.uab.es/?ch=8
    * 2019 : http://icdar2019.org/

* AIhub dataset
    * 한국어 글자체 이미지 : https://aihub.or.kr/aidata/133
    * 다양한 형태의 한글 문자 OCR : https://aihub.or.kr/aidata/33987
    * 야외 실제 촬영 한글 이미지 : https://aihub.or.kr/aidata/33985

* Albumentation
    * [https://github.com/ultralytics/yolov5](https://albumentations.ai/)
