import os
MODEL_PATH = os.path.join('.','runs','detect','train2','weights','best.pt')


from ultralytics import YOLO
import cv2
import math
def liveDetect(): 
    
    f_types = [('Mp4 Files', '*.mp4'),('Avi Files','*.avi'),('*','*')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    cap=cv2.VideoCapture(filename)

    frame_width=int(cap.get(3))
    frame_height = int(cap.get(4))

    out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 100, (frame_width, frame_height))

    model=YOLO(MODEL_PATH)
    classNames = ["fire","default","smoke"]
    while True:
        success, img = cap.read()
        # Doing detections using YOLOv8 frame by frame
        #stream = True will use the generator and it is more efficient than normal
        results=model(img,stream=True)
        #Once we have the results we can check for individual bounding boxes and see how well it performs
        # Once we have have the results we will loop through them and we will have the bouning boxes for each of the result
        # we will loop through each of the bouning box
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                #print(x1, y1, x2, y2)
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                print(x1,y1,x2,y2)
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
                #print(box.conf[0])
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                #print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
        out.write(img)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF==ord('1'):
            break
    out.release()



import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.filedialog import askopenfile
import tkinter.font as font


root = tk.Tk()

root.geometry("200x200") # set the root dimensions
root.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
root.resizable(0, 0) # makes the root window fixed in size.

myFont = font.Font(family='Helvetica')

frame1 = tk.LabelFrame(root,text='Live video Detection')
frame1.place(height=150, width=200 , rely=0.05 ,relx=0)

label = tk.Label(frame1 , text='Note exit button is 1')
label.place(rely=0.2,relx=0.2)

button_dec = tk.Button(frame1 , text = "Start the live detection" , command = lambda :liveDetect())
button_dec.place(rely=0.6,relx=0.2)


root.mainloop()