from evaluate import load

mse = load("mse")
results = mse.compute(references=[6.4, 1.9, 3.5, 2.8, 9.7], \
    predictions=[5.9, 2.3, 4.1, 2.7, 8.6], squared=False)
print(results)