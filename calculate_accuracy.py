import sys

def get_XY(line, prefix, i):
    if line.startswith(prefix):
        # breakpoint()
        return line[-1]
    else:
        return ""
    
def get_BN(line, prefix, i):
    if line.startswith(prefix):
        # Neither or Both가 있으면 가져오기, 아니면 ""
        pass
    else:
        return ""

if __name__ == '__main__':
    fname = sys.argv[1] # "fewshotCoT_500.txt"
    first_n = sys.argv[2] if len(sys.argv) > 2 else None

    with open(fname, "r") as f:
        lines = [line[:-1] for line in f.readlines()] # remove \n

    # pr_bn_value = "".join([get_BN(line, "pred_before : ", i) for i, line in enumerate(lines)]) # only for _rest

    pr_value = "".join([get_XY(line, "pred : ", i) for i, line in enumerate(lines)])
    gt_value = "".join([get_XY(line, "GT : ", i) for i, line in enumerate(lines)])

    # breakpoint()

    assert(len(pr_value) == len(gt_value))
    length = len(pr_value)

    if first_n is not None:
        pr_value = pr_value[:int(first_n)]
        gt_value = gt_value[:int(first_n)]
        length = int(first_n)
    
    n_correct = sum([1 if pr_value[i] == gt_value[i] else 0 for i in range(length)])
    n_XY_correct = sum([1 if pr_value[i] == gt_value[i] and pr_value[i] in ["X", "Y"] else 0 for i in range(length)])
    acc = n_correct / length
    XY_acc = n_XY_correct / length

    print(f"Accuracy = {acc*100:.4f}")
    print(f"XY Accuracy = {XY_acc*100:.4f}")
    print(f"정답 개수:", n_correct)

    # breakpoint()


'''

both 및 neither 제외하고,  X Y 경우만 체크했을 때 
'''