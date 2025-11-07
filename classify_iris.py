from sklearn.datasets import load_iris

iris_dataset = load_iris()

print(f"Keys of iris_dataset: {iris_dataset.keys()}")

# Brief description
print(iris_dataset['DESCR'][:193] + "\n...")

# Properties (width etc)
print(iris_dataset['feature_names'])

# Specie names
print(iris_dataset['target_names'])

# Specie name encoded in numbers 0-2
print(iris_dataset['target'])
print(iris_dataset['data'][:3])
print(f"Data shape of iris_dataset: {iris_dataset['data'].shape}")