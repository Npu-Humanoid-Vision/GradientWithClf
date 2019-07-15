import os
import shutil
import random

def GetImgPaths(folder_path):
    paths = []
    for fpathe,dirs,fs in os.walk(folder_path):
        for f in fs:
            paths.append(f)
    return paths


if __name__ == "__main__":
    root_path = "./Rotated/"
    dst_path = "./"
    data_type = ["Test/", "Train/"]
    sub_paths = ["Pos/", "Neg/"]

    test_set_rate = 0.25
    paths = []

    for i in sub_paths:
        paths.append(GetImgPaths(root_path+i))
    
    # 抽取测试集
    for i in range(len(paths)):
        target_sum = len(paths[i])*(1-test_set_rate)
        while len(paths[i]) > target_sum:
            lucky_dog = random.choice(range(len(paths[i])))# lucky dog to be test data
            shutil.copyfile(root_path+sub_paths[i]+paths[i][lucky_dog], dst_path+data_type[0]+sub_paths[i]+paths[i][lucky_dog])
            print(root_path+sub_paths[i]+paths[i][lucky_dog])
            print(dst_path+data_type[0]+sub_paths[i]+paths[i][lucky_dog])
            print()
            del paths[i][lucky_dog]
    for i in range(len(paths)):
        for lucky_dog in range(len(paths[i])):# lucky dog to be train data
            shutil.copyfile(root_path+sub_paths[i]+paths[i][lucky_dog], dst_path+data_type[1]+sub_paths[i]+paths[i][lucky_dog])
            print(root_path+sub_paths[i]+paths[i][lucky_dog])
            print(dst_path+data_type[1]+sub_paths[i]+paths[i][lucky_dog])
            print()