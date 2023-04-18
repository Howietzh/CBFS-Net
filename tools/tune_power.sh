nohup python tools/train.py my_config/CBBSNet/exp1.py --work-dir tune_power/4 --seed 0 --gpu-id 1 --cfg-options lr_config.power=4 &
nohup python tools/train.py my_config/CBBSNet/exp1.py --work-dir tune_power/6 --seed 0 --gpu-id 2 --cfg-options lr_config.power=6 &
nohup python tools/train.py my_config/CBBSNet/exp1.py --work-dir tune_power/8 --seed 0 --gpu-id 3 --cfg-options lr_config.power=8 &
nohup python tools/train.py my_config/CBBSNet/exp1.py --work-dir tune_power/10 --seed 0 --gpu-id 4 --cfg-options lr_config.power=10 &
nohup python tools/train.py my_config/CBBSNet/exp1.py --work-dir tune_power/15 --seed 0 --gpu-id 5 --cfg-options lr_config.power=15 &
nohup python tools/train.py my_config/CBBSNet/exp1.py --work-dir tune_power/20 --seed 0 --gpu-id 6 --cfg-options lr_config.power=20 &
nohup python tools/train.py my_config/CBBSNet/exp1.py --work-dir tune_power/40 --seed 0 --gpu-id 7 --cfg-options lr_config.power=40 &