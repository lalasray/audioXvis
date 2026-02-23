import string
import cv2
import numpy as np

import argparse


def click_and_crop(event, x, y, flags, param):
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    global pt_list, mouse_click
    if event == cv2.EVENT_LBUTTONDOWN:
        pts_list[idx][mouse_click][0] = x
        pts_list[idx][mouse_click][1] = y 
        
        print("Mouse click "+ str(mouse_click+1)+" added.")
        
        show_img_pts(img,pts)
        mouse_click+=1
        if mouse_click>=5:
            mouse_click=0
        
def show_img_pts(img, pts):
    frame = img.copy()
    for i , pt in enumerate(pts):
        if not np.all(pt == 0):
            frame = cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (255,255,210-i*50), -1)
    cv2.imshow("image", frame)



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_video", required = True, help = "Path to video")
ap.add_argument("-o", "--output_csv", required = True, help = "Path to video")
ap.add_argument("-f", "--fps", required = True, type=int, help = "frame rate <= 30")
args = vars(ap.parse_args())

vid_path =args["input_video"]
cap = cv2.VideoCapture(vid_path)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total video frames: "+str(frame_count))
fps = cap.get(cv2.CAP_PROP_FPS)
print("Original Video FPS: "+ str(fps))
skip_frame = fps/args["fps"]

frame_list = []
n_frame = 0

#frame_count = 120 #for cdebugging

# load video into buffer
while n_frame < frame_count:
    ret, frame = cap.read()
    if not ret:
        break
    if n_frame % skip_frame == 0:
        print(f"Loading video to buffer, plz wait: {n_frame/frame_count:.2%} \r", end='')
        frame = frame[0:600, 600:1200] #crop to us video
        #frame = cv2.resize(frame, (500,500))
        frame_list.append(frame)
    n_frame += 1

print()
print(len(frame_list))

pts_list = np.zeros((len(frame_list),5,2))
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
idx = 0
mouse_click = 0
while True:
    mouse_click = 0
    pts = pts_list[idx]
    img = frame_list[idx]
    print("Frame "+str(idx)+": ")
    show_img_pts(img,pts)

    # wait for a keypress
    key = cv2.waitKey(0)
    if key == ord("q"):
        break
    if key == ord("d"):  #next frame
        if idx <len(frame_list)-1: idx +=1
    if key == ord("a"):  #previsou frame
        if idx >0: idx -=1
    if key == ord("c"):  #clear points
        pts_list[idx] = np.zeros((5,2))
    if key == ord("s"):
        np.savetxt(args["output_csv"],pts_list.reshape(-1,10) , delimiter=",")
        print("File saved to output dir.")
        break


# close all open windows
cv2.destroyAllWindows() 