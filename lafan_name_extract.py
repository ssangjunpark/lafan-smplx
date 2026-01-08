from pathlib import Path


path = Path("/home/sangjunpark/Desktop/lafan_smplx")

files = sorted(path.glob("*.npz"))
names = [p.name.split('.')[0] for p in files]

print(names)

for name in names:
    print(name)
