import ctypes

ll = ctypes.cdll.LoadLibrary
lib = ll("./libsumcall.so")
lib.sum(1, 3)
print('***finish***')
