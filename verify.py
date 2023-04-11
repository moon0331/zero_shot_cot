from collections import Counter
from pprint import pprint
import sys

zeroshotcot = bool(int(sys.argv[1]))
restriction = bool(int(sys.argv[2]))

dir1 = 'restriction' if restriction else 'norestriction'
dir2 = 'zeroshotcot' if zeroshotcot else 'zeroshot'

file_dir = f'result/{dir1}/{dir2}'

def get_filename(is_zeroshotcot, idx, restriction):
    fname = ''
    fname += 'zeroshotcot' if is_zeroshotcot else 'zeroshot'
    fname += f'{idx}'
    fname += '_rest' if restriction else ''
    fname += '.txt'
    return fname

files = [f'{file_dir}/{get_filename(zeroshotcot, i, restriction)}' for i in range(12)]

pprint(files)

for file in files:
    with open(file, 'r') as f:
        lines = f.readlines()
        result_line = [line.strip() for line in lines if line.startswith('pred_before :')]
        cnt = Counter(result_line)
        print(file)
        pprint(cnt)
        # breakpoint()

'''
답변 Counter를 보여주고
해당 답변과 실제 답변(X, Y, B, N)을 매핑한 후
답변 순서대로 XYBN string을 출력 (50개, 500개, 3000개)
'''