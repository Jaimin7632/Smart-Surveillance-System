import sys
import pickle
import struct
import json
import socket
import h5py
import face1 as face1
from PIL import Image
import tensorflow as tf
import numpy as np
import face_recognition
from sklearn.manifold import TSNE
import threading
import os
import time
import six.moves.urllib as urllib
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO
import cv2
from handle import *
sys.path.append("..")
import warnings
# Suppress LabelEncoder warning
warnings.filterwarnings('ignore')

from utils import label_map_util

from utils import visualization_utils as vis_util

from db_conn import *

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score

print("Do You Want Train Data? (Y/N) : ")
ans_user = input()
if ans_user=="y" or ans_user == "Y":
    face1.create_embedded()
    print("Data are Trained...")


global_lock = threading.Lock()
knn1 = KNeighborsClassifier(n_neighbors=5, weights='distance')

svc1 = svm.SVC(kernel='rbf', gamma=1, C=1.0, probability=True)
encoder1 =None
mic = {}

thresh = 0.25
from sklearn.metrics import f1_score, accuracy_score


def pre_load():
    global knn1, svc1,encoder1,mic
    metadata = face1.load_metadata('images')
    num = len(metadata)
    filename = 'tempFiles/embedded2.hdf5'
    f = h5py.File(filename, 'r')
    
    for raw in f:
        for j in f[raw]:
            xc = j.split('.')
            mic[str(raw),int(xc[0])] = f[raw][j][:]    
    embedded2 = np.zeros((metadata.shape[0], 128))
    for i, m in enumerate(metadata):
        embedded2[i] = f[m.name+"/"+m.file].value

    
    f.close() 


    train_in =np.ones((metadata.shape[0]), dtype=bool)
    test_in =np.ones((metadata.shape[0]), dtype=bool)
    name=None
    for i,m in enumerate(metadata):    # 90 / 10 dataset 
        if name==None:
            name=m.name
            train_in[i]=False
            test_in[i]=True
            continue
        if name==m.name:
            train_in[i]=True
            test_in[i]=False
        else:
            name=m.name
            train_in[i]=False
            test_in[i]=True
    distances = [] # squared L2 distance between pairs
    identical = [] # 1 if same identity, 0 otherwise

    num = len(metadata)

    for i in range(num - 1):
        for j in range(1, num):
            distances.append(face1.distance(embedded2[i], embedded2[j]))
            identical.append(1 if metadata[i].name == metadata[j].name else 0)
            
    distances = np.array(distances)
    identical = np.array(identical)

    thresholds = np.arange(0.2, 0.4, 0.005)

    #print(thresholds)
    # for t in thresholds:
    #     print(identical)
    #     print(f1_score(identical, distances < t))
    f1_scores = [f1_score(identical, distances < t) for t in thresholds]
    acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

    opt_idx = np.argmax(f1_scores)
    # Threshold at maximal F1 score
    opt_tau = thresholds[opt_idx]
    # Accuracy at maximal F1 score
    opt_acc = accuracy_score(identical, distances < opt_tau)
    global thresh
    thresh = opt_tau+0.05
    print(f'Accuracy at threshold {opt_tau:.3f} = {opt_acc:.3f}')


    #####################################


    targets2 = np.array([m.name for m in metadata])

    encoder1 = LabelEncoder()
    encoder1.fit(targets2)

    y = encoder1.transform(targets2)


    # 50 train examples of 10 identities (5 examples each)
    X_train = embedded2[train_in]
    # 50 test examples of 10 identities (5 examples each)
    X_test = embedded2[test_in]

    y_train = y[train_in]
    y_test = y[test_in]



    knn1.fit(X_train, y_train)
    svc1.fit(X_train, y_train)

    acc_knn1 = accuracy_score(y_test, knn1.predict(X_test))
    acc_svc1 = accuracy_score(y_test, svc1.predict(X_test))
    print(f'KNN accuracy = {acc_knn1}, SVM accuracy = {acc_svc1}')
    
pre_load()
##############
past_time = None
def retrain():
        now = datetime.datetime.now()
        th =threading.current_thread()
        global past_time
        if past_time == None:
                past_time = now
        time_diff =int((now-past_time).total_seconds())/60
        import platform
        if platform.system() == 'Windows':
            win =1
        else:
            win =0

        if time_diff >3:
                global main_c
                past_time = now
                while global_lock.locked():
                    continue
                global_lock.acquire()
                print('New Face adding...')
                main_c.execute("select * from add_person")
                data = main_c.fetchall()
                main_c.execute("truncate table add_person")
                f = h5py.File('tempFiles/temp.hdf5','a')
                g = h5py.File('tempFiles/embedded2.hdf5','a')
                for d in data:
                        if 1:
                            iemd = f[d[1]].value
                            del f[d[1]] 
                            gg = g.create_group(d[2])
                            
                            if win !=1:
                                os.system("mkdir images/"+str(d[2]))
                                os.system("mv temp_image/"+str(d[1])+" images/"+str(d[2])+"/1.jpg")
                            else:
                                os.system("md images\\"+str(d[2]))
                                os.system("move temp_image\\"+str(d[1])+" images\\"+str(d[2])+"\\1.jpg")

                            no_img=1
                            gg.create_dataset(str(no_img)+".jpg", data=iemd)
                            for key in f.keys():
                                if face1.distance(f[key].value,iemd) < 0.30:
                                    
                                    main_c.execute('delete from notification where  img_name=%s',(key))
                                    no_img=no_img+1
                                    if win !=1:
                                        os.system("mv temp_image/"+str(key)+" images/"+str(d[2])+"/"+str(no_img)+".jpg")
                                    else:
                                        os.system("move temp_image\\"+str(key)+" images\\"+str(d[2])+"\\"+str(no_img)+".jpg")
                                    gg.create_dataset(str(no_img)+".jpg", data=f[key].value)
                                    
                                    del f[key]
                                    
                            while no_img<3:
                                    nn = no_img+1
                                    if win !=1:
                                        os.system("cp images/"+str(d[2])+"/"+str(no_img)+".jpg images/"+str(d[2])+"/"+str(nn)+".jpg")
                                    else:
                                        os.system("copy images\\"+str(d[2])+"\\"+str(no_img)+".jpg images\\"+str(d[2])+"\\"+str(nn)+".jpg")
                                    no_img = no_img+1
                                    gg.create_dataset(str(no_img)+".jpg", data=iemd)
                            main_c.execute('insert into user_detail(name,no_img,last_update_date) values(%s,%s,%s)',(d[2],no_img,now.strftime("%Y-%m-%d")))
                            db.commit()
                            
                        # except Exception as e:
                        #     print(e)
                # main_c.execute("select * from for_asking where flag = 1")
                # data1 = main_c.fetchall()
                # main_c.execute("delete from for_asking where flag=1")
                # for d1 in data1:
                #         local_c.execute("select no_img from user_detail where name = %s",(d1[1]))
                #         no_img = local_c.fetchone()
                #         no = no_img[0] + 1
                #         sf = len(os.listdir(os.pajoin(path, d1[1])))
                #         s =int(sf)+1
                #         local_c.execute("update user_detail set no_img = %s",(no))
                #         os.system("mv temp_image/"+str(d1[2])+" images/"+str(d1[1])+"/"+s+".jpg")
                f.close()
                g.close()
                global_lock.release()
                print('### complete face adding ###')                #retrain model
                if not len(data) == 0:
                    pre_load()

                
                #retrain model
                #retrain model
                


################



def testing_run(frame):
    global thresh
    
    if 1:
        img= frame
        face_locations = face_recognition.face_locations(img)
        #print("I found {} face(s) in this photograph.".format(len(face_locations)))
        if len(face_locations) ==0 :
            print("noface")
            #data_maintain("noface",img,np.array(np.random(1,128)))
        else:    
            for face_location in face_locations:

                # Print the location of each face in this image
                top, right, bottom, left = face_location
				# You can access the actual face itself like this:
                face_image = img[top:bottom, left:right]
                #cv2.rectangle(frame, (left-5, top-5), (right-5, bottom+5), (255,0,0), 2)
                
            
                example_image = np.asarray(face_image)

                fd = face_recognition.face_encodings(example_image)[0]
                example_prediction1 = knn1.predict([fd])

                example_identity1 = encoder1.inverse_transform(example_prediction1)[0]
                results = face_recognition.compare_faces([mic[(example_identity1,1)]], fd)

				
                if face1.distance(mic[(example_identity1,1)],fd)<thresh and face1.distance(mic[(example_identity1,2)],fd) < thresh:
                    print(example_identity1)
                    data_maintain(example_identity1, face_image,fd)
                   
                # elif face1.distance(mic[(example_identity1,0)],fd) < thresh2 and face1.distance(mic[(example_identity1,1)],fd) < thresh2 :
                #     print(example_identity1)
                #     print("2")
                #     for_asking(example_identity1, face_image,fd)
                else:
                    print("unknown")
                    data_maintain("unknown", face_image,fd)
            
   
    # except Exception as e:
    #      print(e)

###################################

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90





tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')




label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


fb = h5py.File('tempFiles/temp.hdf5','a')

def write_to_file(img_name,emd):
        while global_lock.locked():
                continue

        global_lock.acquire()
        print('#### writing ###')
        fb.create_dataset(img_name, data=emd)
        global_lock.release()



class myThread1(threading.Thread):
    def __init__(self, image):
        threading.Thread.__init__(self)
        self.time = time
        self.image = image
        self.per = None
        # self.db = PyMySQL.connect("localhost","root","","main_db" )
        # self.db1=PyMySQL.connect("localhost","root","","local_db" )
        # self.main_c = self.db.cursor()
        # self.local_c=self.db1.cursor()
        self.img_n = None
        self.temd =None
      
 
        
    def run(self):
        self.per = detection(self.image.copy())
        testing_run(self.image)
        retrain()
        # self.db.commit()
        # self.db1.commit()
        # self.main_c.close()
        # self.local_c.close()
        # self.db.close()
        # self.db1.close()
        if not self.img_n ==None:
            write_to_file(self.img_n,self.temd)

        
         
        
            






def detection(img):
    image_np = img
    image_np_expanded = np.expand_dims(image_np, axis=0)
    #print(type(image_np_expanded))
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
    image_np,name1 = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return name1


in_upload = False
clientsocket=None
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
past2 = datetime.datetime.now()
def go_live(frame):
    global in_upload,clientsocket,encode_param,past2
    if in_upload:
        now = datetime.datetime.now()
        result, frame = cv2.imencode('.jpg', frame, encode_param)
        data = pickle.dumps(frame, 0)
        size = len(data)
        clientsocket.sendall(struct.pack(">L", size) + data)
        print("upload ")
        if ((now-past2).total_seconds() / 60.0) > 0.30:
            print("2min #########")
            main_c.execute("UPDATE variable set value='0' where name='live'");
            db.commit()
            in_upload= False
            clientsocket.close()
            
        
    else:
        main_c.execute("select value from variable where name='live'")
        ll = main_c.fetchone()
        db.commit()
        if ll[0]=='1':
            in_upload=True
            past2 = datetime.datetime.now()
            clientsocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            clientsocket.connect(('localhost',8089))









cam1 = cv2.VideoCapture(0)
sess = tf.Session(graph=detection_graph)
mpast =datetime.datetime.now()
while True:
    #try:
        #clean_process()
        _,img = cam1.read()
        img2 =img.copy()
        now = datetime.datetime.now()
        if ((now-mpast).total_seconds()) > 0.30:
            mpast=now
            th1 = myThread1(img2)
            th1.start()
            
        #go_live(img)
        
    # except Exception as e:
    #     print(e)
    #     break
    




db1.close()
db.close()
