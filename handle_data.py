import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import medmnist
from medmnist import INFO, Evaluator

def get_mnist_dataset(batch_size=10, data_flag = 'pathmnist'):
    import medmnist
    from medmnist import INFO, Evaluator
    print(medmnist.__version__)

    download = True
    info = INFO[data_flag]
    task = info['task']
    DataClass = getattr(medmnist, info['python_class'])
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=True)
    eval_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=True)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=True)


    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_loader = data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2 * batch_size, shuffle=False, drop_last=True)

    return train_loader, eval_loader, test_loader
