import hashlib
model = "zenless-lab/sdxl-anything-xl"
print(hashlib.sha256(model.encode('utf-8')).hexdigest())
