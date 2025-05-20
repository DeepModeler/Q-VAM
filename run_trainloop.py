import os

all_command = ['deepspeed --include=localhost:1,2,3 trainloop.py --model_type stablelm --preK 4',
               'deepspeed --include=localhost:1,2,3 trainloop.py --model_type phi --preK 4',
               'deepspeed --include=localhost:1,2,3 trainloop.py --model_type stablelm --preK 8',
               'deepspeed --include=localhost:1,2,3 trainloop.py --model_type phi --preK 8',
               'deepspeed --include=localhost:1,2,3 trainloop.py --model_type stablelm --preK 12',
               'deepspeed --include=localhost:1,2,3 trainloop.py --model_type phi --preK 12']

for the_command in all_command:
    os.system(the_command)