from datagenerate import generate_data

def k_fold_indices(n, k):
    indices = list(range(n))
    fold_sizes = [n//k + 1 if i < n%k else n//k for i in range(k)]
    current = 0
    result = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = indices[:start] + indices[stop:]
        result.append((train_indices, test_indices))
        current = stop
    return result

data = generate_data(100,5)
t = data['t']
mi = data['mi']
kfold = k_fold_indices(100, 10)
a, b = kfold[0]
select = [t[i] for i in b]
s = mi[a]
print(select)