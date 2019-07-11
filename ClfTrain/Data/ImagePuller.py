import os
import datetime
import cv2

def GetImgs(root_path_in_phone, save_path_in_pc):

    date = datetime.datetime.now().strftime("%Y%m%d")
    date = '20190710'

    result = os.popen("adb shell ls "+ root_path_in_phone).read()
    result_list = [i for i in result.split() if i != '' and date in i]
    for i in result_list:
        os.system("adb pull " + root_path_in_phone+i + save_path_in_pc)

    return 

if __name__ == "__main__":
    GetImgs("/storage/emulated/0/DCIM/Camera/", " ./Raw/")   
