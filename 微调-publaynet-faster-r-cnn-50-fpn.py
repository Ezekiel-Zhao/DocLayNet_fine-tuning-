

# 环境安装
import subprocess

# 安装 layoutparser
subprocess.run(["pip", "install", "layoutparser"])
subprocess.run(["pip", "install", "layoutparser[layoutmodels]"])
subprocess.run(["pip", "install", "layoutparser[ocr]"])

# 安装其他库
subprocess.run(["pip", "install", "tesseract"])
subprocess.run(["pip", "install", "imagesize"])
subprocess.run(["pip", "install", "layoutparser", "torchvision"])
subprocess.run(["pip", "install", "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"])

# 克隆 layout-model-training 仓库
subprocess.run(["git", "clone", "https://github.com/Layout-Parser/layout-model-training.git"])


class CFG:
    # 更改配置文件

    cfg_weights_path = '/kaggle/working/model_pre_train.pth' #需要微调的模型路径
    cfg_max_iteration = '43131'       # 大概三个epoch，以Batch_szie = 2为例
    cfg_img_batch_size = '2'          # Batch_size
    cfg_check_point = '10000'          # 每10000次保存一次模型
    cfg_eval_period = '10000'          # 每10000次评估一次模型效果
    cfg_output_path = "/kaggle/working/layout-model-training/configs/prima/model_config.yaml"  #模型配置文件的保存路径
    cfg_NUM_GPU = '1'  #GPU数量
    cfg_git_clone_path = "/kaggle/working/layout-model-training" # GitHub 下载路径

    train_input_path = '/kaggle/input/doclaynet/COCO/train.json' # 原json文件路径
    test_input_path = '/kaggle/input/doclaynet/COCO/test.json'
    
    output_train_path = '/kaggle/working/train.json'  # 清洗后的json文件路径
    output_test_path = '/kaggle/working/test.json'
    
    img_path = '/kaggle/input/doclaynet/PNG'  # 图像地址
    
    max_num_each_category = 5000 # 每个种类最多需要的照片数量

    outout_dir = '/kaggle/working/'

import os
# 切换工作目录到 layout-model-training/
os.chdir(CFG.cfg_git_clone_path)


import layoutparser as lp

# 下载模型
import requests
model_url = "https://www.dropbox.com/s/dgy9c10wykk4lq4/model_final.pth?dl=1"

# 发起HTTP请求以下载文件
response = requests.get(model_url)

# 获取文件名
file_name = "model_pre_train.pth"

# 将下载的内容保存到本地文件
with open(file_name, "wb") as file:
    file.write(response.content)
    
    
# 加载基于PubLayNet训练好的faster-RCnn-50-FPN
pre_trained_model = lp.Detectron2LayoutModel("lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

# 下载对应configuration
import yaml

# 获得配置信息并更改
config_dict = pre_trained_model.cfg
config_dict['MODEL']['WEIGHTS'] = CFG.cfg_weights_path
config_dict['SOLVER']['MAX_ITER'] = CFG.cfg_max_iteration
config_dict['SOLVER']['IMS_PER_BATCH'] = CFG.cfg_img_batch_size
config_dict['SOLVER']['CHECKPOINT_PERIOD'] = CFG.cfg_check_point
config_dict['TEST']['EVAL_PERIOD'] = CFG.cfg_eval_period

# 保存为 YAML 文件
output_file = CFG.cfg_output_path
with open(output_file, "w") as f:
    yaml.dump(config_dict, f)

print(f"Configuration saved to {output_file}")
del pre_trained_model




# 数据清理

import json

def clean_the_label(file_path, output_path, each_category_max_num = 5000):
    '''
    input: file_path: the json file contains label information
           output_path: the path to save the cleaned JSON file
    output: cleaned json file which contains 4 categories of label
    type of data is a dictionary of list of dictionary i.e {key1 : [dict1, dict2, ...], key2: [dict1, dict2, ...], ...}
    dict_keys(['categories', 'images', 'annotations'])
    '''
    
    # Get the data
    with open(file_path, 'r') as file:
        data = json.load(file)
        
        
    '''
    STEP 1: 筛选数据集
    思路：遍历data['image']中的所有img, 获取其种类并计数，大于临界值就不收录在新建的list中
         然后返回新list，并将data['image']改为新list
         最后，根据新的image list我们对annotation做筛选
    '''    
    # total num of each category cant excess each_category_max_num
    count = {
             'financial_reports': 0,
             'scientific_articles' : 0,
             'government_tenders' : 0,
             'laws_and_regulations' : 0,
             'manuals' : 0,
             'patents' : 0
            }
    # 筛选各个种类前each_category_max_num张
    new_img_info = []
    for img_info in data['images']:
        if count[img_info['doc_category']] < each_category_max_num:
            count[img_info['doc_category']] += 1
            new_img_info.append(img_info)
            
    # Set data['images'] = new_img_info
    data['images'] = new_img_info
    
    # Get the image_id and store in a set
    id_set = set()
    for item in data['images']:
        id_set.add(item['id'])
    
    # 筛选annotation
    new_annotation_info = []
    for ann_info in data['annotations']:
        if ann_info['image_id'] in id_set:
            new_annotation_info.append(ann_info)
            
    data['annotations'] = new_annotation_info
        
    
    '''
    STEP2: 筛选label
    '''
    annotations = data['annotations']
    old2new = {
              10: 0, 1: 0, 2: 0, 4:0,
               7: 3, 3: 3, 7: 3,
               8: 1, 6: 1, 11: 1,
               9: 2,
               }
    
    # Filter out entries with category_id == 5 (page-footer)
    cleaned_data = [d for d in annotations if d['category_id'] != 5]

    for idx, dic in enumerate(cleaned_data):
        cleaned_data[idx]['category_id'] = old2new.get(dic['category_id'])
    
    data['annotations'] = cleaned_data
    
    
    # rename Picture and List-item
    
    new_category = [{'supercategory': 'Figure', 'id': 3, 'name': 'Figure'},
                    {'supercategory': 'Table', 'id': 2, 'name': 'Table'},
                    {'supercategory': 'Text', 'id': 0, 'name': 'Text'},
                    {'supercategory': 'Title', 'id': 1, 'name': 'Title'}]
    
    data['categories'] = new_category
        
    # Save the cleaned data to a new JSON file
    with open(output_path, 'w') as output_file:
        json.dump(data, output_file)
        



# Clean the label data and save it to the output path
print('label-cleaning started')
clean_the_label(CFG.test_input_path, CFG.output_test_path, CFG.max_num_each_category)
clean_the_label(CFG.train_input_path, CFG.output_train_path, CFG.max_num_each_category)
print('Done')



# 切换工作目录到 layout-model-training/tools
os.chdir(CFG.cfg_git_clone_path + '/tools')
# 构造运行命令和参数的列表
command = [
    "python", "train_net.py",
    "--num-gpus", CFG.cfg_NUM_GPU,
    "--dataset_name", "Doclayout",
    "--json_annotation_train", CFG.output_train_path,
    "--image_path_train", CFG.img_path,
    "--json_annotation_val", CFG.output_test_path,
    "--image_path_val", CFG.img_path,
    "--config-file",CFG.cfg_output_path,
    "OUTPUT_DIR", CFG.outout_dir
]

# 运行命令
subprocess.run(command)
