import datetime
import cv2
import ftplib
from db_conn import *
import os
import time
import threading
import json
import h5py
import requests
filter_time = None
filter_array = []
def filter(name):
        global filter_array
        global filter_time
        th =threading.current_thread()
        now = datetime.datetime.now()
        dt =now.strftime("%Y-%m-%d")
        time  =now.strftime("%H:%M")
        cstr  = '{"activity":"'+name+'","time":"'+time+'","date":"'+dt+'"}'
        if filter_time == None:
                filter_time = now
        time_diff =int((now-filter_time).total_seconds())/60
        if time_diff > 10:
                filter_array = []
                filter_time = now
        if cstr not in filter_array:
                filter_array.append(cstr)
                print("diffrent #######")
                return False
        else:
                print("repeat #######")
                return True;

      

        
def upload_img(now,imu,emd):
        th =threading.current_thread()
        img_name = str(now.strftime("%Y%m%d_%H%M%S"))+".jpg"
        cv2.imwrite("temp_image/"+img_name,imu)
        cv2.imwrite("temp_image/thumbnail/"+img_name,cv2.resize(imu.copy(),(120,120)))
        file = {'image': open("temp_image/"+img_name,'rb')}                  # file to send
        r= requests.post(server_url+'/receive_file.php', files=file,data={'location': '/'})
        file = {'image': open("temp_image/thumbnail/"+img_name,'rb')}                  # file to send
        r= requests.post(server_url+'/receive_file.php', files=file,data={'location': '/thumbnail'})
        th.img_n= img_name
        th.temd=emd

        
def data_maintain(nam, im,emd):
        th =threading.current_thread()
        sper =th.per
        if filter(nam+sper):
                return None
        now = datetime.datetime.now()
        dt =now.strftime("%Y-%m-%d")
        time  =now.strftime("%H:%M")
        img_name = str(now.strftime("%Y%m%d_%H%M%S"))+".jpg"
        r= requests.post(server_url+'/handle_data.php',data={'name': nam,'person':sper,'date':dt,'time':time,'img_name':img_name})
        print(r.text)
        if r.text == "1":
                upload_img(now,im,emd)
