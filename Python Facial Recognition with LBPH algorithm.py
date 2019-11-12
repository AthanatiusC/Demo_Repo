import cv2
import time
import threading
import multiprocessing
import queue
import os
import numpy as np
import pickle
import sys
import system_logging as log
from datetime import datetime
import psutil
import redis
import imutils
import send
from flask_opencv_streamer.streamer import Streamer
# import vlib

os.system('cls' if os.name=='nt' else 'clear')
log.sys("Initializing System Core")

class VCORE:
    def __init__(self):
        continue

    def check_core(self):
        try:
            if not os.path.isfile(self.core):
                self.create_model()
            self.recognizer_model.read(self.core)
        except:
            pass

    def check_cam(self):
        online_cams = []
        for i in range(psutil.cpu_count(logical=True)):
            vid = cv2.VideoCapture(i)
            if vid.isOpened():
                online_cams.append(i)
        return online_cams

    def create_model(self):
        try:
            
            self.prepare_training_dataset("dataset")
            file = pickle.load(open(self.pickles, "rb"))
            label = file["labels"]  # DECENTRIALIZE DICTIONARY
            face = file["faces"]
            self.recognizer_model.train(face, np.array(label))
            self.recognizer_model.write(self.core)
        except Exception as ex:
            print(ex)

    def update_model(self):
        file = pickle.load(open(self.pickles, "rb"))
        label = file["labels"]  # DECENTRIALIZE DICTIONARY
        face = file["faces"]
        self.recognizer_model.read(self.core)
        self.recognizer_model.update(face, np.array(label))
        self.recognizer_model.write(self.core)

    def draw_rectangle(self, img, rect):
        (startX, startY, endX, endY) = rect.astype("int")
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

    def draw_text(self, img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.5, (0, 255, 0), 2)

    def createCLAHE(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
        res = clahe.apply(frame)
        return res

    def new_save(self):
        # WRITE FILE IF RESIST FILE DOESN'T EXIST
        if not os.path.isfile(self.pickles):
            data = {"faces": self.faces, "labels": self.labels}
            file = open(self.pickles, "wb")
            file.write(pickle.dumps(data))
            file.close()

    def save_existing(self):
        newlabels = self.labels
        newfaces = self.faces
        self.predictions.clear()
        self.faces.clear()
        if os.path.isfile(self.pickles):
            oldfaces = []
            oldlabels = [] # LOAD EXISTING RESIST FILE
            file = pickle.load(open(self.pickles, "rb"))
            label = file["labels"]  # DECENTRIALIZE DICTIONARY
            face = file["faces"]
            for la, fa in zip(label, face):  # APPENDING EACH LIST
                oldlabels.append(la)
                oldfaces.append(fa)
            for la, fa in zip(newlabels, newfaces):
                oldlabels.append(la)
                oldfaces.append(fa)
            data = {"faces": oldfaces, "labels": oldlabels}
            files = open(self.pickles, "wb")
            files.write(pickle.dumps(data))
            files.close()
            file = pickle.load(open(self.pickles, "rb"))

    def Serialize(self, frame, box):
        try:
            (startX, startY, endX, endY) = box.astype("int")
            gray = self.createCLAHE(frame)
            equalized = cv2.resize(gray[startY-20:endY+20, startX-20:endX+20], (300, 300), interpolation=cv2.INTER_LANCZOS4)
            # face_aligned = self.alignment.align(300, equalized, self.alignment.getLargestFaceBoundingBox(equalized), landmarkIndices=self.alignment.INNER_EYES_AND_BOTTOM_LIP)
            try:
                np.mean(equalized)
                self.faces.append(equalized)
                self.boxes.append(box.astype("int"))
                self.good+=1
            except:
                self.bad +=1
                exit()
        except Exception as e:
            self.faces.append(None)
            self.boxes.append(None)

    def Normalize(self, frame):
        try:
            h, w, c = frame.shape
            processes = []
            imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (
                104.0, 177.0, 123.0), swapRB=False, crop=False)  # CONVERT FRAME INTO BLOB FOR DNN INPUT
            self.detector.setInput(imageBlob)
            detections = self.detector.forward() # ITERATE ALL DETECTED FACE
            for i in range(0, detections.shape[2]):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    processes.append(threading.Thread(
                        target=self.Serialize, args=(frame, box)))
            for process in processes:
                process.daemon = True
                process.start()
            if len(processes) ==0:
                prediction.append({"label": "", "confidence": 0,"frame": frame, "face": None})
        except:
            pass

    def prepare_training_dataset(self, dataset):
        dirs = os.listdir(self.dataset)
        for dir_name in dirs:
            if not dir_name.startswith("T"):
                path, dirs, file_name = next(
                    os.walk(os.path.join(self.dataset, dir_name)))
                self.bad = 0
                self.good = 0
                total = 0
                label = int(dir_name)
                fullpath = os.path.join(self.dataset, dir_name)
                for images in file_name:
                    frame = cv2.imread(os.path.join(fullpath, images))
                    h, w, c = frame.shape
                    total += 1
                    sys.stdout.write(
                        "\r [ SYSTEM ] : Current id '{}' - preparing {}/{} Images - BAD : {} GOOD : {}".format(label, total, len(file_name),self.bad,self.good))
                    self.Normalize(frame)
                    for face, box in zip(self.faces, self.boxes):
                        if len(self.faces) != 0:
                            # !!DEFAULT EXPECTED RETURN VALUE
                            self.labels.append(label)
                sys.stdout.write("\n")
                sys.stdout.flush() # APPEND DATASET DIR FILE NAME WITH T MEANING ITS BEEN TRAINED
                # os.rename(os.path.join(self.dataset, dir_name),
                #           os.path.join(self.dataset, "T"+dir_name))
        if not os.path.isfile(self.pickles):
            self.new_save()
        if os.path.isfile(self.pickles):
            self.save_existing()

    def predict(self, face, box, frame):
        try:
            if np.average(box) > 250:
                (startX, startY, endX, endY) = box.astype("int")
                label, confidence = self.recognizer_model.predict(face)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                self.draw_rectangle(frame, box)
                data = {"frame": frame}
                if confidence < 100:
                    self.draw_text(frame, str(label)+" Accu: " +
                                    str(100-int(confidence)+15), startX, y)
                    data.update({"label": label, "confidence": confidence,
                            "frame": frame, "face": frame[startY:endY, startX:endX]})
                    self.predictions.append(data)
                    self.datas.put(data)
                else:
                    self.predictions.append(data) # datas.put(data)
        except:
            pass

    def background(self):
        while self.datas.not_empty:
            data = self.datas.get()
            face = data["face"]
            uid = data["label"]
            if uid in self.absence.keys():
                continue
            cv2.imwrite("icon.png", face)
            self.absence.update({uid:uid,"sent":False})
            while not self.absence["sent"]:
                send.post_attendance({"user_id": uid, "camera_id": "5d4c276616171e2938004c72"}, {"photo": open("icon.png", "rb")})

    def store_frame(self, vid):
        while True:
            ret, frame = vid.read()
            if not ret:
                log.sys("Camera disconnected or doesn't online!")
                self.closed = True
            self.frames.append(frame)

    def get_frame(self, cam):
        vid = cv2.VideoCapture(0)
        # vid = cv2.VideoCapture("../SVM/JOHN WICK 3  5 Minute Trailers (4K ULTRA HD) NEW 2019.mkv")
        thread = threading.Thread(target=self.store_frame(vid))
        thread.daemon = True
        thread.start()

    def main(self, cam):
        t = threading.Thread(target=self.get_frame, args=[cam])
        t.daemon = True
        t.start()
        thread = threading.Thread(target=self.background)
        thread.daemon = True
        thread.start()
        while not self.closed:
            while len(self.frames) != 0:
                start = time.time()
                if len(self.frames) == 0:
                    break
                frame = self.frames[0]
                self.frames.clear()
                frame = imutils.resize(frame, width=1080)
                original = frame.copy()
                avg = np.average(frame)
                try:
                    if avg < 130:
                        invGamma = 1.0 / (avg/35)
                        table = np.array([((i / 255.0) ** invGamma) *
                                        255 for i in np.arange(0, 256)]).astype("uint8")
                        frame = cv2.LUT(frame, table)
                except:
                    pass
                self.Normalize(frame)
                if len(self.faces) != 0:
                    threads = []
                    for face, box in zip(self.faces, self.boxes):
                        t = threading.Thread(
                            target=self.predict, args=(face, box, frame))
                        threads.append(t)
                    self.faces.clear()
                    self.boxes.clear()
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    for prediction in self.predictions:
                        frame = prediction["frame"]
                try:
                    frm = "FPS : " + str(int(1.0 / (time.time() - start)))
                    cpu = "CPU : "+str(psutil.cpu_percent()) + "%"
                    mem = "MEM : " + str(psutil.virtual_memory()[2])+"%"
                    self.draw_text(frame, frm, 0, 100)
                    self.draw_text(frame, cpu, 0, 120)
                    self.draw_text(frame, mem, 0, 140)
                    # cv2.imshow("Frame", frame)
                    self.streamer.update_frame(frame)
                    if not self.streamer.is_streaming:
                        self.streamer.start_streaming()
                except Exception as ex:
                    pass
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.closed = True
                    cv2.destroyAllWindows()
                    break
