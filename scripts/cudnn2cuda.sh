#cp /media/dell/data/installpkgs/cudnn-10.2-linux-x64-v8.2.0.53/cuda/include/cudnn*.h /usr/local/cuda/include/
#cp /media/dell/data/installpkgs/cudnn-10.2-linux-x64-v8.2.0.53/cuda/lib64/libcudnn* /usr/local/cuda/lib64/


cp /media/dell/data/installpkgs/cudnn-10.2-linux-x64-v7.6.5.32/cuda/include/cudnn*.h /usr/local/cuda/include/
cp /media/dell/data/installpkgs/cudnn-10.2-linux-x64-v7.6.5.32/cuda/lib64/libcudnn* /usr/local/cuda/lib64/

#cp /media/dell/data/installpkgs/cudnn-10.2-linux-x64-v8.1.0.77/cuda/include/* /usr/local/cuda/include/
#cp /media/dell/data/installpkgs/cudnn-10.2-linux-x64-v8.1.0.77/cuda/lib64/* /usr/local/cuda/lib64/

chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
cat /usr/local/cuda/version.txt
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

