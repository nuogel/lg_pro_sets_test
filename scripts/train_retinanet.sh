python train.py --type obd --model retinanet --ep 100 --cp 0  --bz 16 --lr 0.01 --nw 16 --ema 0 --autoamp 0 --pt 1

#python train.py --type OBD --ep 801 --model yolov5  --cp 'saved/checkpoint/yolov5_voc_68.pkl'  --bz 16 --nw 8 --ema 1 --autoamp 1 --to 1