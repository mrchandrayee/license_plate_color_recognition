import cv2
import numpy as np
from PIL import Image

def _pil_to_bgr(image):
    return cv2.cvtColor(np.array(image.convert("RGB")),cv2.COLOR_RGB2BGR)

def _color_name(bgr):
    """Estimate dominant vehicle body color from HSV pixels."""
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    mask=(v>45)&(s<240)
    if np.count_nonzero(mask)<50: return "Unknown"
    hh=h[mask]; ss=s[mask]; vv=v[mask]
    sat=float(np.median(ss)); val=float(np.median(vv))
    if val<65: return "Black"
    if sat<28:
        if val>190: return "White"
        if val>115: return "Silver"
        return "Grey"
    hue=float(np.median(hh))
    if hue<8 or hue>=170: return "Red"
    if hue<22: return "Orange"
    if hue<35: return "Yellow"
    if hue<85: return "Green"
    if hue<130: return "Blue"
    return "Purple"

def _plate_candidates(bgr):
    """Heuristic plate candidate detector; returns best crop or None."""
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    gray=cv2.bilateralFilter(gray,11,17,17)
    edges=cv2.Canny(gray,50,180)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(17,3))
    closed=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel)
    contours,_=cv2.findContours(closed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    h,w=gray.shape
    candidates=[]
    for c in contours:
        x,y,cw,ch=cv2.boundingRect(c)
        area=cw*ch
        ratio=cw/max(ch,1)
        if 2.0<=ratio<=6.5 and area>w*h*0.0005 and cw>w*0.08 and ch>8:
            candidates.append((area,x,y,cw,ch))
    candidates.sort(reverse=True)
    for _,x,y,cw,ch in candidates[:8]:
        pad=max(3,int(ch*.15))
        crop=bgr[max(0,y-pad):min(h,y+ch+pad),max(0,x-pad):min(w,x+cw+pad)]
        if crop.size: return cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
    return None

def _read_plate(plate_crop):
    if plate_crop is None: return "Not recognized","No plate candidate"
    try:
        import pytesseract
        img=cv2.cvtColor(plate_crop,cv2.COLOR_RGB2GRAY)
        img=cv2.resize(img,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        img=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        text=pytesseract.image_to_string(img,config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        text="".join(ch for ch in text.upper() if ch.isalnum())
        if len(text)>=5: return text,"Tesseract OCR (verify manually)"
        return "Not recognized","OCR text too short"
    except Exception:
        return "Not recognized","OCR unavailable; install Tesseract and pytesseract"

def detect_vehicle_boxes(pil_image):
    """Optional YOLO detection. Returns list of boxes, or [] if unavailable."""
    try:
        from ultralytics import YOLO
        model=YOLO("yolov8n.pt")
        arr=_pil_to_bgr(pil_image)
        result=model.predict(arr,verbose=False,conf=0.25)[0]
        names=model.names
        allowed={"car","motorcycle","bus","truck"}
        boxes=[]
        for box in result.boxes:
            cls=int(box.cls[0])
            if names.get(cls) in allowed:
                boxes.append(tuple(map(int,box.xyxy[0].tolist())))
        return boxes
    except Exception:
        return []

def crop_vehicle(pil_image,box):
    return pil_image.crop(box)

def analyze_image(pil_image,use_yolo=False):
    bgr=_pil_to_bgr(pil_image)
    vehicle="Vehicle (uploaded image)"
    confidence=0.0
    if use_yolo:
        try:
            from ultralytics import YOLO
            model=YOLO("yolov8n.pt")
            result=model.predict(bgr,verbose=False,conf=0.25)[0]
            allowed={"car","motorcycle","bus","truck"}
            found=[]
            for box in result.boxes:
                cls=int(box.cls[0]); name=model.names.get(cls,"object")
                if name in allowed:
                    found.append((float(box.conf[0]),name,tuple(map(int,box.xyxy[0].tolist()))))
            if found:
                confidence,name,coords=max(found,key=lambda x:x[0])
                x1,y1,x2,y2=coords
                bgr=bgr[max(0,y1):max(y1+1,y2),max(0,x1):max(x1+1,x2)]
                vehicle=name.title()
                confidence*=100
        except Exception:
            vehicle="YOLO unavailable (full image used)"
    color=_color_name(bgr)
    plate_crop=_plate_candidates(bgr)
    plate,method=_read_plate(plate_crop)
    return {"plate":plate,"color":color,"vehicle":vehicle,
            "confidence":confidence if confidence else (65.0 if plate_crop is not None else 0.0),
            "plate_method":method,"plate_crop":plate_crop}
