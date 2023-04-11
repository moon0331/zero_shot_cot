import argparse
import logging
import torch
import random
import time
import os
from utils import *

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)
    
    # print("OPENAI_API_KEY:")
    # print(os.getenv("OPENAI_API_KEY"))

    if args.key_num == 1:
        api_key = 
    elif args.key_num == 2:
        api_key = 

    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args, api_key)
    
    print("setup data loader ...")
    dataloader = setup_data_loader(args, False)
    print_now()
    
    # few shot 어떻게 문장 선정하는지 S-BERT 활용하여 융합
    if args.method.startswith('few_shot') and not args.method.startswith('few_shot_cot'): # == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method.startswith('few_shot_cot'): # == "few_shot_cot":
        demo = create_demo_text(args, cot_flag=True)
    else:
        pass
    
    total = 0
    correct_list = []        
    for i, data in enumerate(dataloader):
        if i+1 < args.start_idx: continue # default 1: 1st data 부터 시작하도록 함 (test 필요)
        print('*************************')
        print("{}st data".format(i+1))
                
        # Prepare question template ... ()
        x, y = data
        x = "Q: " + x[0] + "\n" + "A:"
        y = y[0].strip()
        
        if args.method.startswith('zero_shot') and not args.method.startswith('zero_shot_cot'): # == "zero_shot":
            x = x + " " + args.direct_answer_trigger_for_zeroshot
        elif args.method.startswith('zero_shot_cot'): # == "zero_shot_cot":
            x = x + " " + args.cot_trigger
        elif args.method.startswith('few_shot') and not args.method.startswith('few_shot_cot'): # == "few_shot":
            x = demo + x
        elif args.method.startswith('few_shot_cot'): # == "few_shot_cot":
            x = demo + x
        else:
            raise ValueError("method is not properly defined ...")
        
        # Answer prediction by generating text ...
        max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
        z = decoder.decode(args, x, max_length, i, 1)

        # Answer extraction for zero-shot-cot ...
        if args.method.startswith('zero_shot_cot'): # == "zero_shot_cot":
            z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
            max_length = args.max_length_direct
            pred = decoder.decode(args, z2, max_length, i, 2)
            print(z2 + ' ' + pred) # 미관상
        else:
            pred = z
            print(x + ' ' + pred) # 미관상

        # Clensing of predicted answer ...
        pred = answer_cleansing(args, pred) # Hypothesis 2 << 이런거 처리 어떻게 할지
        
        print("pred : {}".format(pred))
        print("GT : " + y)
        print('*************************')
        
        # Checking answer ...
        correct = (np.array([pred]) == np.array([y])).sum().item()
        correct_list.append(correct)
        total += 1 #np.array([y]).size(0)
        
        if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
            break
            #raise ValueError("Stop !!")
    
    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument(
        "--api_log_file_name", type=str, default=None, help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )
    
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    
    parser.add_argument(
        "--dataset", type=str, default="anli_small", 
            choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", 
                     "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters", 
                     "anli", "anli_small", "anli_sample", "anli_dev"], 
            help="dataset used for experiment"
    )
    
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")
    
    parser.add_argument(
        "--model", type=str, default="gpt3.5", choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl", "gpt3.5"], help="model used for decoding. Note that 'gpt3' are the smallest models."
    )
    
    parser.add_argument(
        "--method", type=str, default="zero_shot_cot", choices=[
        "zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot",
        "zero_shot_two_line", "zero_shot_cot_two_line", "few_shot_two_line", "few_shot_cot_two_line",
        "zero_shot_xy", "zero_shot_cot_xy", "few_shot_xy", "few_shot_cot_xy",
        # "zero_shot_0", "zero_shot_cot_0",
        # "zero_shot_1", "zero_shot_cot_1",
        ] + [f'zero_shot_{i}' for i in range(16)] + [f'zero_shot_cot_{i}' for i in range(16)], 
        help="method"
    )

    parser.add_argument(
        "--key_num", type=int, default=1, help="key number. 1 for DIAL(Default), 2 for Individual"
    )

    parser.add_argument(
        "--add_rest", action='store_true', help="whether to add restriction (X, Y, Both, Neither). Default: False"
    )

    parser.add_argument(
        "--cot_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    ) # default: 128
    parser.add_argument(
        "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=10, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.2, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    parser.add_argument(
        "--start_idx", type=int, default=1, help="start index of test dataset"
    )
    parser.add_argument(
        "--n_few_shot_example", type=int, default=8, help="number of few-shot examples"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "anli": # test set 전체
        args.dataset_path = "./dataset/aNLI/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer is" ########## 수정 필요할수 있음 (second prompt)
    elif args.dataset == "anli_small": # 500개
        args.dataset_path = "./dataset/aNLI/test_sample.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer is" ########## 수정 필요할수 있음 (second prompt)
    elif args.dataset == "anli_sample": # fewshot을 위한 임시 답변 생성
        print('anli_sample is for few-shot sampling.')
        args.dataset_path = "./dataset/aNLI/fewshot_candidate.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer is" ########## 수정 필요할수 있음 (second prompt)
    elif args.dataset == "anli_dev": # dev set 전체
        args.dataset_path = "./dataset/aNLI/dev.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer is" ########## 수정 필요할수 있음 (second prompt)
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    
    args.direct_answer_trigger_for_fewshot = "The answer is"
    
    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."
    elif args.cot_trigger_no == 2:
        args.cot_trigger = "We should think about this step by step."
    elif args.cot_trigger_no == 3:
        args.cot_trigger = "First,"
    elif args.cot_trigger_no == 4:
        args.cot_trigger = "Before we dive into the answer,"
    elif args.cot_trigger_no == 5:
        args.cot_trigger = "Proof followed by the answer."
    elif args.cot_trigger_no == 6:
        args.cot_trigger = "Let's think step by step in a realistic way."
    elif args.cot_trigger_no == 7:
        args.cot_trigger = "Let's think step by step using common sense and knowledge."
    elif args.cot_trigger_no == 8:
        args.cot_trigger = "Let's think like a detective step by step."
    elif args.cot_trigger_no == 9:
        args.cot_trigger = "Let's think about this logically."
    elif args.cot_trigger_no == 10:
        args.cot_trigger = "Let's think step by step. First,"
    elif args.cot_trigger_no == 11:
        args.cot_trigger = "Let's think"
    elif args.cot_trigger_no == 12:
        args.cot_trigger = "Let's solve this problem by splitting it into steps."
    elif args.cot_trigger_no == 13:
        args.cot_trigger = "The answer is after the proof."
    elif args.cot_trigger_no == 14:
        args.cot_trigger = "Let's be realistic and think step by step."
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")
    
    return args

if __name__ == "__main__":
    main()