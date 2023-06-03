

CUDA_VISIBLE_DEVICES=1 \
python train.py --scheduler_step_size 20  --batch_size 16  --model_name new_54_01E-4 --png --data_path /home/ljy/KITTI \
--load_weights_folder /home/ljy/DIFFNet-main__copy/res/new_428_7E-4/30042023-01:35:02/models/weights_9 \
--learning_rate 1e-5 \

CUDA_VISIBLE_DEVICES=1 \
python train.py --scheduler_step_size 20  --batch_size 16  --model_name new_54_9E-4 --png --data_path /home/ljy/KITTI \
--load_weights_folder /home/ljy/DIFFNet-main__copy/res/new_428_7E-4/30042023-01:35:02/models/weights_9 \
--learning_rate 0.9e-4 \

CUDA_VISIBLE_DEVICES=1 \
python train.py --scheduler_step_size 20  --batch_size 16  --model_name new_54_8E-4 --png --data_path /home/ljy/KITTI \
--load_weights_folder /home/ljy/DIFFNet-main__copy/res/new_428_7E-4/30042023-01:35:02/models/weights_9 \
--learning_rate 0.8e-4 \

CUDA_VISIBLE_DEVICES=1 \
python train.py --scheduler_step_size 20 --batch_size 16  --model_name new_54_7E-4 --png --data_path /home/ljy/KITTI \
--load_weights_folder /home/ljy/DIFFNet-main__copy/res/new_428_7E-4/30042023-01:35:02/models/weights_9 \
--learning_rate 0.7e-4 \

CUDA_VISIBLE_DEVICES=1 \
python train.py --scheduler_step_size 20  --batch_size 16  --model_name new_54_6E-4 --png --data_path /home/ljy/KITTI \
--load_weights_folder /home/ljy/DIFFNet-main__copy/res/new_428_7E-4/30042023-01:35:02/models/weights_9 \
--learning_rate 0.6e-4 \

CUDA_VISIBLE_DEVICES=1 \
python train.py --scheduler_step_size 20  --batch_size 16  --model_name new_54_5E-4 --png --data_path /home/ljy/KITTI \
--load_weights_folder /home/ljy/DIFFNet-main__copy/res/new_428_7E-4/30042023-01:35:02/models/weights_9 \
--learning_rate 0.5e-4 \

# CUDA_VISIBLE_DEVICES=1 \
# python train.py --scheduler_step_size 20 --batch_size 16  --model_name new_428_4E-4 --png --data_path /home/ljy/KITTI \
# --load_weights_folder /home/ljy/DIFFNet-main__copy/res/new_428_1E-4/28042023-19:50:51/models/weights_6 \
# --learning_rate 0.4e-4 \

CUDA_VISIBLE_DEVICES=1 \
python train.py --scheduler_step_size 20 --batch_size 16  --model_name new_54_4E-4 --png --data_path /home/ljy/KITTI \
--load_weights_folder /home/ljy/DIFFNet-main__copy/res/new_428_7E-4/30042023-01:35:02/models/weights_9 \
--learning_rate 0.4e-4 \

CUDA_VISIBLE_DEVICES=1 \ 
python train.py --scheduler_step_size 20 --batch_size 16  --model_name new_54_3E-4 --png --data_path /home/ljy/KITTI \
--load_weights_folder /home/ljy/DIFFNet-main__copy/res/new_428_7E-4/30042023-01:35:02/models/weights_9 \
--learning_rate 0.3e-4 \
   
CUDA_VISIBLE_DEVICES=1 \
python train.py --scheduler_step_size 20 --batch_size 16  --model_name new_54_2E-4 --png --data_path /home/ljy/KITTI \
--load_weights_folder /home/ljy/DIFFNet-main__copy/res/new_428_7E-4/30042023-01:35:02/models/weights_9 \
--learning_rate 0.2e-4 \











# CUDA_VISIBLE_DEVICES=1 \
# python train.py --scheduler_step_size 14  --batch_size 16  --model_name ASPP_NO_LOSS_CA --png --data_path /home/ljy/KITTI \
# --load_weights_folder /home/ljy/DIFFNet-main__copy/res/ASPP_fina_sim/22112022-19:43:12/models/weights_10
