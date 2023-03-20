import sys

def get_line(line, prefix, i):
    return line[-1] if line.startswith(prefix) else "" # X or Y

if __name__ == '__main__':
    fname = sys.argv[1] # "fewshotCoT_500.txt"
    with open(fname, "r") as f:
        lines = [line[:-1] for line in f.readlines()] # remove \n
        pr_value = "".join([get_line(line, "pred : ", i) for i, line in enumerate(lines)])
        gt_value = "".join([get_line(line, "GT : ", i) for i, line in enumerate(lines)])

        assert(len(pr_value) == len(gt_value))

        length = len(pr_value)

        # breakpoint()

        
        acc = sum([1 if pr_value[i] == gt_value[i] else 0 for i in range(length)]) / length

        print(f"{acc*100:.4f}")

        # breakpoint()


'''

zeroshot_500.txt -> 74.6 (0.18달러)
zeroshotCoT_500.txt -> 77.8 ..
fewshot_500.txt -> 77.6 | 78.2
fewshotCoT_500 -> 76.6

'''