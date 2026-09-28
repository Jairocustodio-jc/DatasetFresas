from ultralytics import YOLO
m=YOLO("yolov8n.pt")
m.train(data="ds/data.yaml",epochs=60,imgsz=640,batch=16,device="cpu",workers=4,patience=15,project="runs",name="fresas",exist_ok=True,
        plots=False,verbose=False,fliplr=0.5,mosaic=1.0,hsv_h=0.0,hsv_s=0.3,hsv_v=0.3)   # sin variar el tono: el color ES la clase
