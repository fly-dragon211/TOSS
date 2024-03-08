source activate hf-torch

cd ../..
langs=(ruby javascript php go java python)
#langs=(javascript php go)
#langs=(python)

code_length=256
eval_batch_size=256



for lang in ${langs[*]}
do
   CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --code_length ${code_length} --lang ${lang} \
   --online_cal --device "cuda" --num_workers 40 \
   --eval_batch_size ${eval_batch_size} --run_function "main_coclr_multi_stage1_allLang"
done
