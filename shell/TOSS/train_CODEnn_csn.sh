cd ../..
langs=(ruby python javascript php go java)
nl_length=80
code_length=232
eval_batch_size=128
num_train_epochs=10
CUDA_VISIBLE_DEVICES=("0,1,2,3" )

for lang in ${langs[*]}
do

CUDA_VISIBLE_DEVICE=${CUDA_VISIBLE_DEVICES[device_index]}
if [[ `expr $device_index + 1` -ge ${#CUDA_VISIBLE_DEVICES[*]} ]]
then
  wait
  fi

let device_index="($device_index + 1) % ${#CUDA_VISIBLE_DEVICES[*]}"
echo ${CUDA_VISIBLE_DEVICE}

train_batch_size=64
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICE} python train_Graphcodebert.py  \
--num_workers 40 --train_batch_size ${train_batch_size} --TrainModel CODEnn \
--device cuda --learning_rate 2e-5 --lang ${lang} \
--seed 5  --nl_length ${nl_length} --ClassifyLoss "bce" \
--code_length ${code_length} --eval_batch_size ${eval_batch_size} \
--num_train_epochs ${num_train_epochs}

#sleep $(python -c 'import random; print(random.randint(0,1))') & paralle

# torch.distributed.launch --nproc_per_node=8  --master_port 29555
done