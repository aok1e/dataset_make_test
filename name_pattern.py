import os.path
import time
from datetime import datetime

def patterned_filename(subdir,op):
    
    out_date = datetime.now().strftime("%Y%m%d_%H%M")
    
    #ディレクトリとファイル名の対応（ここ要改善）
    dict = {
         "labeled_img":"labeled_",
         "labeled_txt":"",
         "result_train":"result_",
         "result_model":"model_",
         "result_inspection":"inspection_",
         "result_loss_graph":"loss_",
         "result_predict_txt":"predict_inspection_",
         "result_real_time_estimation":"real_time_estimation_",
    }
    
    filename = dict[subdir] + out_date

    #1つ上のディレクトリ取得
    hand_thumb =  os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    
def add_datetime():
    
    out_date = datetime.now().strftime("%Y%m%d_%H%M")

    return out_date

if __name__ == "__main__":
    print(add_datetime())
    