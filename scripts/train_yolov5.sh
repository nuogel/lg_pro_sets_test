python train.py --type obd --model yolov5 --ep 10 --cp 1  --bz 8 --lr 0.00005 --nw 16 --ema 1 --autoamp 1 --pt 1

#python train.py --type OBD --ep 801 --model yolov5  --cp 'saved/checkpoint/yolov5_voc_68.pkl'  --bz 16 --nw 8 --ema 1 --autoamp 1 --to 1