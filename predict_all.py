import os
from ultralytics import YOLO

model = YOLO("D:/rice_project/runs/classify/train/weights/best.pt")

test_dir = "D:/rice_project/Rice/Rice/test"

class_map = {
    "Rice__Brown_Spot": "Brown_Spot",
    "Rice__Healthy": "Healthy",
    "Rice__Leaf_Blast": "Leaf_Blast",
    "Rice__Neck_Blast": "Neck_Blast"
}

for folder_name, true_class in class_map.items():
    class_path = test_dir + "/" + folder_name
    if not os.path.isdir(class_path):
        print(f"⚠️ 文件夹不存在: {class_path}")
        continue
    
    print(f"\n📂 正在预测类别: {true_class}")
    results = model.predict(source=class_path, save=True, verbose=False)
    
    correct = 0
    total = 0
    for r in results:
        pred_class = r.names[r.probs.top1]
        total += 1
        if pred_class == true_class:
            correct += 1
    
    print(f"✅ 准确率: {correct}/{total} = {correct/total*100:.1f}%")