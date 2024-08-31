import cv2

def vid_to_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True :
        success,img = cap.read()
        if not success:
            break
        frames.append(img)
    return frames

def frames_to_vid(frames,path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path,fourcc,30,(frames[0].shape[1],frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()

def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)

