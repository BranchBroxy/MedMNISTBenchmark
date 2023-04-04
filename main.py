import torch

def init_function():
    # Use a breakpoint in the code line below to debug your script.
    print(torch.cuda.current_device())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    init_function()
    from models import ResNet, ResidualBlock, SparseResNet, SparseResidualBlock
    resnet18 = ResNet(ResidualBlock, [2, 2, 2, 2])
    resnet34 = ResNet(ResidualBlock, [3, 4, 6, 3])
    in_features, out_features = 3 * 28 * 28, 9
    sparseresnet34 = SparseResNet(in_features, out_features, SparseResidualBlock, [3, 4, 6, 3])

    from handle_data import get_mnist_dataset
    train_loader, eval_loader, test_loader = get_mnist_dataset(batch_size=128, data_flag="dermamnist")

    from compare_models import compare_models_acc_over_epoch, compare_models_robustness
    # compare_models_acc_over_epoch(train_loader, eval_loader, test_loader, resnet18, resnet34, sparseresnet34)
    compare_models_robustness(train_loader, eval_loader, test_loader, resnet18, resnet34, sparseresnet34)


