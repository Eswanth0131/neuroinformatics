Subset of `mofei` — dataset + smp_unet usage

Overview


Key references
- Dataset loader: [mofei/dataset.py](mofei/dataset.py)
- Model implementation: [mofei/models/smp_unet.py](mofei/models/smp_unet.py)

How to load the dataset (example)
- The `mofei` folder contains dataset utilities — import and instantiate the dataset class defined there. The exact class name may vary; inspect `mofei/dataset.py` to use the correct name.

Example Python snippet (adjust class/name args to match `mofei/dataset.py`):

```python
# example_load_dataset.py
from torch.utils.data import DataLoader
import mofei.dataset as ds_module

# Replace `DatasetClass` with the actual class name in mofei/dataset.py
DatasetClass = getattr(ds_module, 'Dataset', None) or getattr(ds_module, 'MRIDataset', None)
if DatasetClass is None:
    raise RuntimeError('Open mofei/dataset.py and use the real dataset class name')

dataset = DatasetClass(root='path/to/data', split='train')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
for batch in dataloader:
    images, targets = batch
    print(images.shape)
    break
```

Notes
- Inspect `mofei/dataset.py` and `mofei/models/smp_unet.py` for exact class/function names and arguments.
- If you want, I can create small wrapper scripts (`run_train.sh`, `quick_test_model.py`) here that match the exact class names found in `mofei`; tell me and I will auto-generate them.

Files added here
- [mofei_subset/README.md](mofei_subset/README.md)

Done.
