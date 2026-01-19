import hashlib
s = "cagliostrolab/animagine-xl-4.0"
print(hashlib.sha256(s.encode('utf-8')).hexdigest())
