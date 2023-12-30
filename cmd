/mnt/16t/xzwnlp/direct-preference-optimization/.cache/xzwnlp/llama2-7b-base-safety_2023-12-27_10-31-29_818864/LATEST/policy.pt

python -u train.py loss=dpo loss.beta=0.1 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 \
    model.archive=/mnt/16t/xzwnlp/direct-preference-optimization/.cache/xzwnlp/llama2-7b-base-safety-plus_2023-12-29_22-08-02_000533/LATEST/policy.pt

python -u train.py loss=dpo loss.beta=0.1 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 \
    model.archive=/mnt/16t/xzwnlp/direct-preference-optimization/.cache/xzwnlp/llama2-7b-base-safety-plus-sft_2023-12-30_19-51-53_326023/LATEST/policy.pt

python -u train.py loss=dpo loss.beta=0.1 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 \
    model.archive=/mnt/16t/xzwnlp/direct-preference-optimization/.cache/xzwnlp/llama2-13b-chat-safety-plus-sft_2023-12-30_22-59-14_061324/LATEST/policy.pt

python -u train.py loss=dpo loss.beta=0.1 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 \
    model.archive=/mnt/16t/xzwnlp/direct-preference-optimization/.cache/xzwnlp/llama2-13b-chat-safety-plus-sft_2023-12-30_22-59-14_061324/LATEST/policy.pt
  