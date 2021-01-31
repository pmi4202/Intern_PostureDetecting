# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000
# source venv/bin/activate

# import the necessary packages
from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import math
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model

outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# vs = VideoStream(src=0).start()

time.sleep(2.0)

tic=0
# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

#[start]yerin
states = list(range(4)) #good posture, arm, kneel, back
#[end]yerin

@app.route("/")
def index():
    load_state()
    # return the rendered template
    return render_template("index.html", states=states)

#[start]yerin
def load_state():
    states[0]="posture"

    file = open ("armstate.txt", "r")
    states[1] = file.read()
    file.close
    
    file = open ("kneelstate.txt", "r")
    states[2] = file.read()
    file.close
    
    file = open ("backstate.txt", "r")
    states[3] = file.read()
    file.close

    if(states[1]=="Not Folding Hands" and states[2]=="Not kneeling" and states[3]=="Straight"):
        states[0]="good posture"
    elif(states[1]=="Folding Hands" or states[2]=="kneeling" or states[3]=="Hunchback" or states[3]=="Reclined"):
        states[0]="bad posture"
    else:
        states[0]="Not detected"

    return 0
#[end]yerin

def detect_motion():
    global outputFrame, lock

    cap=cv2.VideoCapture(0)
    cap.set(100,160)
    cap.set(200,120)


    # loop over frames from the video stream
    while True:
        ret,frame=cap.read()
        #test
        frame = imutils.resize(frame, width=800)
        #
        params, model_params = config_reader()

        oriImg = frame # B,G,R order  
        multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
        for m in range(1):
            scale = multiplier[0]
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                            model_params['padValue'])
            input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)
            output_blobs = model.predict(input_img)
            heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
            heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                                interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                    :]
            heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
            paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
            paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                            interpolation=cv2.INTER_CUBIC)
            paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
            heatmap_avg = heatmap_avg + heatmap / len(multiplier)
            paf_avg = paf_avg + paf / len(multiplier)

        all_peaks = [] #To store all the key points which a re detected.
        peak_counter = 0
        
        # prinfTick(1) #prints time required till now.

        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            map = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map.shape)
            map_left[1:, :] = map[:-1, :]
            map_right = np.zeros(map.shape)
            map_right[:-1, :] = map[1:, :]
            map_up = np.zeros(map.shape)
            map_up[:, 1:] = map[:, :-1]
            map_down = np.zeros(map.shape)
            map_down[:, :-1] = map[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        connection_all = []
        special_k = []
        mid_num = 10

        # prinfTick(2) #prints time required till now.

        canvas = frame# B,G,R order
        for i in range(18): #drawing all the detected key points.
            for j in range(len(all_peaks[i])):
                cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
        print()
        position = checkPosition(all_peaks)
        checkKneeling(all_peaks)
        checkHandFold(all_peaks)
        print()
        print()

        with lock:
            outputFrame = canvas.copy()
    cap.release()


def checkPosition(all_peaks):
    #[start]yerin
    file = open ("backstate.txt", "w")
    #[end]yerin
    try:
        f = 0
        if (all_peaks[16]):
            a = all_peaks[16][0][0:2] #Right Ear
            f = 1
        else:
            a = all_peaks[17][0][0:2] #Left Ear
        b = all_peaks[11][0][0:2] # Hip
        angle = calcAngle(a,b)
        degrees = round(math.degrees(angle))
        if (f):
            degrees = 180 - degrees
        if (degrees<70):
            #[start]yerin
            file.write("Hunchback")
            #[end]yerin
            print("Hunchback")
            return 1
        elif (degrees > 110):
            #[start]yerin
            file.write("Reclined")
            #[end]yerin
            print ("Reclined")
            return -1
        else:
            #[start]yerin
            file.write("Straight")
            #[end]yerin
            print("Straight")
            return 0
    except Exception as e:
        #[start]yerin
        file.write("person not in lateral view and unable to detect ears or hip")
        #[end]yerin
        print("person not in lateral view and unable to detect ears or hip")
    #[start]yerin
    file.close()
    #[end]yerin

def calcAngle(a, b):
    try:
        ax, ay = a
        bx, by = b
        if (ax == bx):
            return 1.570796
        return math.atan2(by-ay, bx-ax)
    except Exception as e:
        print("unable to calculate angle")

def checkHandFold(all_peaks):
    #[start]yerin
    file = open ("armstate.txt", "w")
    #[end]yerin
    try:
        if (all_peaks[3][0][0:2]):
            try:
                if (all_peaks[4][0][0:2]):
                    distance  = calcDistance(all_peaks[3][0][0:2],all_peaks[4][0][0:2]) #distance between right arm-joint and right palm.
                    armdist = calcDistance(all_peaks[2][0][0:2], all_peaks[3][0][0:2]) #distance between left arm-joint and left palm.
                    if (distance < (armdist + 100) and distance > (armdist - 100) ): #this value 100 is arbitary. this shall be replaced with a calculation which can adjust to different sizes of people.
                        print("Not Folding Hands")
                        #[start]yerin
                        file.write("Not Folding Hands")
                        #[end]yerin
                    else: 
                        print("Folding Hands")
                        #[start]yerin
                        file.write("Folding Hands")
                        #[end]yerin
            except Exception as e:
                print("Folding Hands")
                #[start]yerin
                file.write("Folding Hands")
                #[end]yerin
    except Exception as e:
        try:
            if(all_peaks[7][0][0:2]):
                distance  = calcDistance( all_peaks[6][0][0:2] ,all_peaks[7][0][0:2])
                armdist = calcDistance(all_peaks[6][0][0:2], all_peaks[5][0][0:2])
                if (distance < (armdist + 100) and distance > (armdist - 100)):
                    print("Not Folding Hands")
                    #[start]yerin
                    file.write("Not Folding Hands")
                    #[end]yerin
                else: 
                    print("Folding Hands")
                    #[start]yerin
                    file.write("Folding Hands")
                    #[end]yerin
        except Exception as e:
            print("Unable to detect arm joints")
            #[start]yerin
            file.write("Unable to detect arm joints")
            #[end]yerin

def calcDistance(a,b): #calculate distance between two points.
    try:
        x1, y1 = a
        x2, y2 = b
        return math.hypot(x2 - x1, y2 - y1)
    except Exception as e:
        print("unable to calculate distance")

def checkKneeling(all_peaks):
    #[start]yerin
    file = open ("kneelstate.txt", "w")
    #[end]yerin
    f = 0
    if (all_peaks[16]):
        f = 1
    try:
        if(all_peaks[10][0][0:2] and all_peaks[13][0][0:2]):
            rightankle = all_peaks[10][0][0:2]
            leftankle = all_peaks[13][0][0:2]
            hip = all_peaks[11][0][0:2]
            leftangle = calcAngle(hip,leftankle)
            leftdegrees = round(math.degrees(leftangle))
            rightangle = calcAngle(hip,rightankle)
            rightdegrees = round(math.degrees(rightangle))
        if (f == 0):
            leftdegrees = 180 - leftdegrees
            rightdegrees = 180 - rightdegrees
        if (leftdegrees > 60  and rightdegrees > 60): # 60 degrees is trail and error value here. We can tweak this accordingly and results will vary.
            print ("Both Legs are in Kneeling")
            #[start]yerin
            file.write("kneeling")
            #[end]yerin
        elif (rightdegrees > 60):
            print ("kneeling")
            #[start]yerin
            file.write("Right leg is kneeling")
            #[end]yerin
        elif (leftdegrees > 60):
            print ("Left leg is kneeling")
            #[start]yerin
            file.write("kneeling")
            #[end]yerin
        else:
            print ("Not kneeling")
            #[start]yerin
            file.write("Not kneeling")
            #[end]yerin

    except IndexError as e:
        try:
            if (f):
                a = all_peaks[10][0][0:2] # if only one leg (right leg) is detected
            else:
                a = all_peaks[13][0][0:2] # if only one leg (left leg) is detected
            b = all_peaks[11][0][0:2] #location of hip
            angle = calcAngle(b,a)
            degrees = round(math.degrees(angle))
            if (f == 0):
                degrees = 180 - degrees
            if (degrees > 60):
                print ("Both Legs Kneeling")
                #[start]yerin
                file.write("kneeling")
                #[end]yerin
            else:
                print("Not Kneeling")
                #[start]yerin
                file.write("Not kneeling")
                #[end]yerin
        except Exception as e:
            print("legs not detected")
            #[start]yerin
            file.write("legs not detected")
            #[end]yerin

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		with lock:
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			if not flag:
				continue

		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())
    
	model = get_testing_model()
	model.load_weights('./model/keras/model.h5')
	
	# while(1):
	# 	ret,frame=cap.read()
	# 	params, model_params = config_reader()
	# 	canvas = detect_motion(frame, params, model_params)    
	# 	cv2.imshow("capture",canvas) 
	# 	if cv2.waitKey(1) & 0xFF==ord('q'):
	# 		break
	# cap.release()
	
	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion)
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
# vs.stop()