# PSFDataset
This is an implementation of the Path Signature Feature methodology for creating datasets for landmark-based human action recognition, following the [paper](https://arxiv.org/abs/1707.03993)
> Developing the Path Signature Methodology and its Application to Landmark-based Human Action Recognition - Weixin Yang, Terry Lyons, Hao Ni, Cordelia Schmid, Lianwen Jin

## Contents
1. [Usage](#usage)
   1. [Short Example](#short-example)
   2. [Transformations](#transformations)
   3. [PSFDataset Class](#psfdataset-class)
   4. [Saving and Loading](#saving-and-loading)
   5. [Use in your Training Pipeline](#use-in-your-training-pipeline)
2. [Requirements](#requirements)

## Usage
### Short Example
The interface is simple and inspired by the transforms and dataset modules of [torchvision](https://pytorch.org/docs/stable/torchvision/index.html). As a simple example: 
```python
transform = transforms.spatial.Normalize()

ds = PSFDataset(transform=transform)
for keypoints,label in data_array:
	ds.add_element(keypoints, label)
```
### Transformations
The usage is inspired by the [torchvision.transforms](https://pytorch.org/docs/stable/torchvision/transforms.html) module. Common transformations to normalise and subsample data as well as the various path transformations for the path signature methodology are implemented as callable transformation objects and can be chained together using Compose.
Initialize a transforms once and then directly apply it to your data (numpy arrays):
```python
# assuming keypoints is a 3D numpy array [frames,landmarks,coords] the following will extract the first 10 frames from the array. 
transform = transforms.spatial.FirstN(10)
keypoints = transform(keypoints)
```

Usually data comes as a spatial path describing the change of the system over time, i.e. a path of the form [frame_id,element,coordinates]. This is the form expected by the spatial transformations. A spatiotemporal path describes the time evolution of single elements, i.e. it is of the form [element,frame_id,coordinates]. This is the input form expected by temporal transformations. Thus between spatial and temporal transformations you have to apply a path transformation to convert the spatial to a spatiotemporal path.
```python
transform = transforms.Compose([
    transforms.spatial.Normalize(),
    transforms.SpatioTemporalPath(),
    transforms.temporal.MultiDelayedTransformation(2)
])
```
Compose is intendend to be used only once in a chain. Recursive use should work as expected for transforming the data but will break the dictionary keeping track of what transformations have been applied. 

### PSFDataset Class
The PSFDataset class provides an easy interface for creating and managing the path signature featureset. Create a transformation or a chain of transformations and pass it into the PSFDataset objects contructor, similar to the [torchvision.datasets](https://pytorch.org/docs/stable/torchvision/datasets.html) objects, to automatically transform any data (in the form of numpy arrays) added to the dataset: 
```python
transform = transforms.Compose([
    transforms.spatial.Crop(),
    transforms.spatial.Normalize(),
    transforms.spatial.Tuples(2),
    transforms.SpatioTemporalPath(),
    transforms.temporal.MultiDelayedTransformation(2),
    transforms.temporal.DyadicPathSignatures(dyadic_levels=1,
                                             signature_level=3)
])

dataset = PSFDataset(transform=transform)
for keypoints,label in data_array:
	dataset.add_element(keypoints, label)
```
PFSDatasets further provide a from_iterator method which expects an iterable returning pairs of keypoints and labels and essentially wraps the for loop at the bottom of the above example.

To combine several PSFDatasets to make up your final feature set (e.g. using both tuples and triples of landmarks as in above paper) create several PSFDataset objects and combine them into one single dataset using PSFZippedDataset. For example:
```python
transform1 = # on chain of transformations
ds1 = PSFDataset(transform=transform1)

transform2 = # another chain of transformations
ds2 = PSFDataset(transform=transform2)

dataset = PSFZippedDataset((ds1, ds2))
```
The PSFZippedDataset class exposes the same interface for accessing the data and using it in your training pipeline as described below. Creating the dataset as well as saving and loading needs to be done on an individual basis.


### Saving and Loading
PSFDatasets allow saving and loading of the dataset by saving the transformed data and labels to a .npz file (indexed as 'data' and 'labels') for easy loading of the data elsewhere, and saving the information of transformations that have been applied to the data into a separate .json file to easily keep track of dataset settings.
```python
data_iterator = ...
transform = ...
ds = PSFDataset(transform)
ds.from_iterator(data_iterator)
# This creates two files 'my_psf_dataset.npz' containing the data and labels
# and 'my_psf_dataset.json' containing the transformation information.
ds.save(filename="my_psf_dataset")

# Load the saved data back from disk into memory
ds2 = PSFDataset()
ds2.load(filename="my_psf_dataset")
```
Loading currently only loads the data, rebuilding the transformation is not yet supported.

Alternatively the data can be loaded from disc using numpy directly
```python
with np.load("my_psf_dataset.npz") as file_content:
  data = file_content['data']
  labels = file_content['labels']
```

### Use in your Training Pipeline
The PSFDataset class is compatible and easy to use with both PyTorch and TensorFlow.
#### PyTorch
PSFDataset implements __getitem__ and __len__ methods and can thus be  passed to a [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). For example:
```python
psf_ds = PSFDataset()
psf_ds.load("my_psf_dataset")
data_loader = torch.utils.data.DataLoader(psf_ds,
                                          batch_size=8,
										  shuffle=True,
                                          num_workers=2)
```
#### TensorFlow
PSFDataset provides a python generator which can be consumed by [tf.data.Dataset.from_generator](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator). It also provides the final feature dimension to define the output shape. For example
```python
psf_ds = PSFDataset()
psf_ds.load("my_psf_dataset")
ds = tf.data.Dataset.from_generator(
    psf_ds.get_iterator, 
    output_types=(tf.float64, tf.int64), 
    output_shapes=((psf_ds.get_data_dimension(),), ())
)
ds = ds.shuffle(buffer_size=100).batch(8)
```

#### TensorBoard
The dictionary returned by get_desc() can be passed into the hparam tracking of TensorBoard for comparing the effect of different dataset configurations on your model performance.

TensorFlow example:
```python
hparam_callback = hp.KerasCallback(logdir, ds.get_desc())
model.fit(ds, epochs=5,  callbacks=[hparam_callback])
```
PyTorch example:
```python
tb_writer = torch.utils.tensorboard.SummaryWriter(filepath)
tb_writer.add_hparams(ds.get_desc(), {})
```

## Requirements
* numpy
* esig
* tqdm
