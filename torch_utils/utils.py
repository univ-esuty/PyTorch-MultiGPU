import torch
import dnnlib


def init_dataset_kwargs(data):
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
    dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
    dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
    dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
    return dataset_kwargs, dataset_obj.name

def compose_model():
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True  # <- enable it when running on docker.
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 26)
    return model