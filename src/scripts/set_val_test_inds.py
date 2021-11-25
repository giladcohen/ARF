import numpy as np
import os

rand_gen = np.random.RandomState(123456)
dataset_dirs = [
    '/tmp/adversarial_robustness/cifar10',
    '/tmp/adversarial_robustness/cifar100',
    '/tmp/adversarial_robustness/svhn',
    '/tmp/adversarial_robustness/tiny_imagenet'
]

for dataset_dir in dataset_dirs:
    os.makedirs(dataset_dir, exist_ok=True)
    test_test_size, test_size = 2500, 10000

    test_test_inds = rand_gen.choice(np.arange(test_size), test_test_size, replace=False)
    test_test_inds.sort()
    test_val_inds = np.asarray([i for i in np.arange(test_size) if i not in test_test_inds])

    np.save(os.path.join(dataset_dir, 'test_test_inds.npy'), test_test_inds)
    np.save(os.path.join(dataset_dir, 'test_val_inds.npy'), test_val_inds)

# Create the mini val/test for boundary attack
for dataset_dir in dataset_dirs:
    test_test_inds = np.load(os.path.join(dataset_dir, 'test_test_inds.npy'))
    test_val_inds = np.load(os.path.join(dataset_dir, 'test_val_inds.npy'))

    mini_test_test_inds = test_test_inds[:250]
    mini_test_val_inds = test_val_inds[:750]

    np.save(os.path.join(dataset_dir, 'mini_test_test_inds.npy'), mini_test_test_inds)
    np.save(os.path.join(dataset_dir, 'mini_test_val_inds.npy'), mini_test_val_inds)

print('done')

