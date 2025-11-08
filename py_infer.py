# pi_infer.py
import json, torch, numpy as np, cv2, sys
IMG_SIZE = 300
MODEL, LABELS = "waste_effb3.ts", "labels.json"
mean = np.array([0.485,0.456,0.406], np.float32)
std  = np.array([0.229,0.224,0.225], np.float32)

def pre_bgr(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32)/255.0; x = (x-mean)/std
    return torch.from_numpy(np.transpose(x,(2,0,1))[None])

labels = json.load(open(LABELS))
model = torch.jit.load(MODEL, map_location="cpu").eval(); torch.set_num_threads(4)

if len(sys.argv)>1:
    frame = cv2.imread(sys.argv[1]); x = pre_bgr(frame)
    with torch.no_grad(): p = torch.softmax(model(x),1).numpy()[0]
    i = int(p.argmax()); print(labels[i], float(p[i]))
else:
    cap = cv2.VideoCapture(0); assert cap.isOpened()
    while True:
        ok, f = cap.read(); 
        if not ok: break
        with torch.no_grad(): p = torch.softmax(model(pre_bgr(f)),1).numpy()[0]
        i = int(p.argmax()); cv2.putText(f, f"{labels[i]} {p[i]:.2f}", (12,30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Pi", f)
        if (cv2.waitKey(1)&255) in (ord('q'),27): break
    cap.release(); cv2.destroyAllWindows()