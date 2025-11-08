# export_effb3_torchscript.py
import json, torch, torch.nn as nn, timm
IMG_SIZE = 300
CKPT = "best_waste_classifier.pth"
OUT_TS, OUT_LABELS = "waste_effb3.ts", "labels.json"

class WasteClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        nf = self.backbone.num_features
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(nf,512), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(512,num_classes))
    def forward(self,x): return self.classifier(self.backbone(x))

ckpt = torch.load(CKPT, map_location="cpu")
class_names = ckpt["class_names"]
model = WasteClassifier("efficientnet_b3", len(class_names), pretrained=False)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()

ex = torch.randn(1,3,IMG_SIZE,IMG_SIZE)
with torch.inference_mode(): traced = torch.jit.trace(model, ex)
traced.save(OUT_TS); json.dump(class_names, open(OUT_LABELS,"w"))
print("Saved", OUT_TS, OUT_LABELS)
