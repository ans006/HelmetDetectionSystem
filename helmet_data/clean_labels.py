import os

label_dirs = ['train/labels', 'valid/labels']
valid_classes = ['0', '1']  # keep only helmet and no_helmet

for folder in label_dirs:
    path = os.path.join('..', 'helmet_data', folder)
    for file in os.listdir(path):
        if file.endswith('.txt'):
            full_path = os.path.join(path, file)
            with open(full_path, 'r') as f:
                lines = f.readlines()
            new_lines = [line for line in lines if line[0] in valid_classes]
            with open(full_path, 'w') as f:
                f.writelines(new_lines)

print("✅ Dataset cleaned! Only helmet and no_helmet labels kept.")

