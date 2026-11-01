raw_dir = r"D:\1stsemester\Machine Learning\assignment 2\dataset"
data_dir = r"D:\1stsemester\Machine Learning\assignment 2\my_dataset"
classes = ["me", "not_me"]
val_ratio = 0.2

random.seed(42)

for cls in classes:
    src = os.path.join(raw_dir, cls)
    files = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(files)
    n_val = int(len(files) * val_ratio)
    for i, f in enumerate(files):
        split = "val" if i < n_val else "train"
        dst = os.path.join(data_dir, split, cls)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))

for split in ["train", "val"]:
    for cls in classes:
        print(split, cls, len(os.listdir(os.path.join(data_dir, split, cls))))
