import os
import os.path as osp
import json
import glob
import argparse


def main(args):
    file_data = dict()
    file_data["images"] = dict()

    # modify each image's json file 
    for name in glob.glob(os.path.join(args.images_path,'*')):
        temp = dict() 

        tmp = name.split('.')
        tmp[-1] = 'json'
        json_name = ".".join(tmp).replace('images','gt')

        with open(json_name,'r', encoding='UTF8') as f:
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

    # save converted ufo format
    with open(args.out_path, 'w') as f:  
        json.dump(file_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', '-i', type=str, default="/opt/ml/input/data/AIHUB_outside_bookcover/images",
                        help='input images path')
    parser.add_argument('--out_path', '-o', type=str, default="/opt/ml/input/data/AIHUB_outside_bookcover/ufo/train.json",
                        help='out_json_path')
    parser.add_argument('--json_path', '-j', type=str, default= '/opt/ml/input/data/AIHUB_outside_bookcover/gt',
                        help='input jsons path')
    args = parser.parse_args()

    main(args)
