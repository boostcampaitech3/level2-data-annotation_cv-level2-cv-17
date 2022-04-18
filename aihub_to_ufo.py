import os
import os.path as osp
import json

file_data = dict()
file_data["images"] = dict()

# you must check this
# image_path에 ai hub 이미지 데이터 경로를 적어주세요.
image_path = '/opt/ml/input/data/AIHUB_outside_sample/images'
file_list = os.listdir(image_path)

# you must check this
# ufo json 저장할 경로 train.json or valid.json
save_ufo_path = '/opt/ml/input/data/AIHUB_outside_sample/ufo/valid.json'

#저장할 사진의 json파일 열기
for name in file_list:
    temp = dict() # temp에 정보들을 넣어서 최종적으로 ufo로 바꿀 예정

    # 확장자가 jpg, JPG 2가지로 구성됨
    j_name = name.replace('.JPG','.json')
    j_name = j_name.replace('.jpg','.json')

    # you must check this
    # gt_path : ai hub json 파일이 있는 경로
    gt_path = osp.join('/opt/ml/input/data/AIHUB_outside_sample/jsons', j_name)
    with open(gt_path, 'r', encoding='UTF8') as f:
        json_data = json.load(f)
        
        temp["img_h"] = json_data["images"][0]["height"]
        temp["img_w"] = json_data["images"][0]["width"]
        temp["words"] = dict()
        
        i = 0
        for ann in json_data["annotations"]:
            if(len(ann["bbox"])!=4):
                continue
            
            x,y,w,h = ann["bbox"]
            if ann["bbox"][0] is None:
                continue 
                
            temp["words"][i] = {
                "transcription":ann["text"],
                "language": ["ko"]}
            
            if ann["text"] == "xxx":
                temp["words"][i]["illegibility"]=True
            else:
                temp["words"][i]["illegibility"]=False
           

            if json_data["metadata"][0]["wordorientation"] == "가로":
                temp["words"][i]["orientation"] = "Horizontal"
            elif json_data["metadata"][0]["wordorientation"] == "세로":
                temp["words"][i]["orientation"] = "Vertical"
            else:
                temp["words"][i]["orientation"] = "Irregular"

            point = [[x,y],
                     [x+w,y],
                     [x+w,y+h],
                     [x,y+h]]
            temp["words"][i]["points"] = point
            temp["words"][i]["word_tags"]=None
            i+=1
         
    file_data["images"][name] = temp


with open(ufo_path, 'w') as f:   # annotation 내용을 원하는 디렉토리에 저장
    json.dump(file_data, f, indent=4)