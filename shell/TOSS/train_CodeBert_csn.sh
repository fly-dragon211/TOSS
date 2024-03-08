cd ../..
langs=(ruby python javascript php go java)
#langs=(python)
nl_length=80
code_length=256
train_batch_size=64
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


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICE} python train_Graphcodebert.py  \
--num_workers 40 --train_batch_size ${train_batch_size} --TrainModel CodeBertP \
--device cuda --learning_rate 2e-5 --lang ${lang} --max_neg --ClassifyLoss bce \
--seed 5 --encoder_name_or_path "microsoft/graphcodebert-base" --nl_length ${nl_length} \
--code_length ${code_length} --eval_batch_size ${eval_batch_size} \
--num_train_epochs ${num_train_epochs} --overwrite


done