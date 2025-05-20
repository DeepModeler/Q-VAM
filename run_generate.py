import os

all_command=[
'CUDA_VISIBLE_DEVICES=1 python generate.py --dataset rad --model_type stablelm --test_mode final --preK 4',
'CUDA_VISIBLE_DEVICES=1 python generate.py --dataset rad --model_type phi --test_mode final --preK 4',
'CUDA_VISIBLE_DEVICES=1 python generate.py --dataset slake --model_type stablelm --test_mode final --preK 4',
'CUDA_VISIBLE_DEVICES=1 python generate.py --dataset slake --model_type phi --test_mode final --preK 4',
'CUDA_VISIBLE_DEVICES=1 python generate.py --dataset pvqa --model_type stablelm --test_mode final --preK 4'
'CUDA_VISIBLE_DEVICES=1 python generate.py --dataset pvqa --model_type phi --test_mode final --preK 4'
]

for the_command in all_command:
    os.system(the_command)