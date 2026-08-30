import sys

data_path = sys.argv[1] if len(sys.argv) > 1 else "combined.txt"

with open(data_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"文件大小: {len(raw_text)} 字符")

paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
print(f"段落总数: {len(paragraphs)}")

print("\n=== 前5个段落 (各前80字) ===")
for i, p in enumerate(paragraphs[:5]):
    print(f"  [{i}] {p[:80]}...")

print("\n=== 后5个段落 (各前80字) ===")
for i, p in enumerate(paragraphs[-5:]):
    print(f"  [{len(paragraphs)-5+i}] {p[:80]}...")

print("\n=== 模拟打乱后训练/验证划分 ===")
import random
random.seed(42)
shuffled = paragraphs.copy()
random.shuffle(shuffled)
split_idx = int(len(shuffled) * 0.8)
train_paras = shuffled[:split_idx]
val_paras = shuffled[split_idx:]

print(f"训练集: {len(train_paras)} 段落")
print(f"验证集: {len(val_paras)} 段落")

print("\n=== 打乱后训练集前5个段落 (各前80字) ===")
for i, p in enumerate(train_paras[:5]):
    print(f"  [{i}] {p[:80]}...")

print("\n=== 打乱后验证集前5个段落 (各前80字) ===")
for i, p in enumerate(val_paras[:5]):
    print(f"  [{i}] {p[:80]}...")

print("\n=== 检查不同小说的段落分布 ===")
novels = {"verdict": 0, "ethan": 0, "innocence": 0, "other": 0}
for p in paragraphs:
    pl = p.lower()
    if "gisburn" in pl or "stroud" in pl or "verdict" in pl:
        novels["verdict"] += 1
    elif "ethan" in pl or "frome" in pl or "starkfield" in pl:
        novels["ethan"] += 1
    elif "newland" in pl or "ellen" in pl or "mingott" in pl or "may welland" in pl:
        novels["innocence"] += 1
    else:
        novels["other"] += 1

for k, v in novels.items():
    print(f"  {k}: {v} 段落 ({v/len(paragraphs)*100:.1f}%)")