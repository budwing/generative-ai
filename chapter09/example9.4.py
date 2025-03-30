from evaluate import load
ref = [9, 3, 2, 1]
pre = [5, 4, 3, 2]
mse = load("mse")
p = load("pearsonr")
sp = load("spearmanr")
mse_res = mse.compute(references=ref, predictions=pre)
p_res = p.compute(references=ref, predictions=pre)
sp_res = sp.compute(references=ref, predictions=pre)
print(mse_res,p_res, sp_res)