# 전체 데이터 수는 3000개이지만, 비용 문제로 1000개만 사용하도록 설정되어 있음 (anli_small)

python main.py --dataset anli_small --model gpt3.5 --method zero_shot --limit_dataset_size 0 >> zeroshot_500.txt
python main.py --dataset anli_small --model gpt3.5 --method zero_shot_cot --limit_dataset_size 0 >> zeroshotCoT_500.txt
python main.py --dataset anli_small --model gpt3.5 --method few_shot --limit_dataset_size 0 >> fewshot_500.txt
python main.py --dataset anli_small --model gpt3.5 --method few_shot_cot --limit_dataset_size 0 >> fewshotCoT_500.txt