import torch
import torch.nn as nn
def compare_models_acc_over_epoch(train_dataloader, eval_dataloader, test_dataloader, *models: nn.Module) -> None:
    """
    Compare the train and eval accuracy over epochs for multiple PyTorch models.

    Args:
    - train_dataloader: PyTorch DataLoader for training dataset
    - eval_dataloader: PyTorch DataLoader for validation dataset
    - test_dataloader: PyTorch DataLoader for testing dataset
    - *models: one or more PyTorch models to compare

    Returns: None

    """
    from handle_model import handle_model
    acc_list_per_noise_level = []
    model_handlers = []
    model_names = []
    for model in models:
        model_handlers.append(handle_model(model, train_dataloader, eval_dataloader, test_dataloader))
        model_names.append(model.__class__.__name__)
        #models[0].__class__.__name__
    train_acc_list = []
    eval_acc_list = []
    train_loss_list = []

    epochs = 100
    learning_rate = 0.001
    for model_runner in model_handlers:
        model_runner.run(epochs=epochs, learning_rate=learning_rate)
        train_acc_list.append(model_runner.train_acc)
        eval_acc_list.append(model_runner.eval_acc)
        train_loss_list.append(model_runner.train_loss)


    path = "Compare"

    import pandas as pd
    df_train_acc = pd.DataFrame(train_acc_list).T
    test_path_save = path + "_train_acc"
    titel = "Train Accuracy comparison of NN"
    plot_acc_df(df_train_acc, model_names, test_path_save, titel)

    df_eval_acc = pd.DataFrame(eval_acc_list).T
    eval_path_save = path + "_eval_acc"
    titel = "Eval Accuracy comparison of NN"
    plot_acc_df(df_eval_acc, model_names, eval_path_save, titel)

    df_train_loss = pd.DataFrame(train_loss_list).T
    train_loss_path_save = path + "_train_loss"
    titel = "Loss comparison of NN"
    plot_loss_df(df_train_loss, model_names, train_loss_path_save, titel)





def plot_acc_df(df, model_names, path_save, titel):
    """
    Plot the accuracy comparison chart for PyTorch models.

    Args:
    - df: Pandas dataframe containing accuracy values
    - model_names: list of model names
    - path_save: path to save the plot
    - titel: plot title

    Returns: None

    """
    df.columns = model_names
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    for model_name in model_names:
        path_save = path_save + "_" + model_name
    # path_save = path_save + "_train_" + ".png"
    path_save = path_save + ".png"
    figure, axes = plt.subplots()
    epochs = len(df.index)
    axes.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if epochs < 10:
        axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
        axes.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.grid()
    plt.ylabel("Accuracy in %")
    plt.xlabel("Epochs")
    plt.title(titel)
    for i in range(df.shape[1]):
        sns.lineplot(data=df, x=df.index + 1, y=df.iloc[:, i], ax=axes, label=df.columns[i], marker="*",
                     markersize=8)

    plt.legend(loc='lower right')
    # axes.legend(labels=["Acc1", "Acc2"])
    plt.savefig(path_save)
    plt.close()

def plot_loss_df(df, model_names, path_save, titel):
    """
    Plot the loss comparison chart for PyTorch models.

    Args:
    - df: Pandas dataframe containing loss values
    - model_names: list of model names
    - path_save: path to save the plot
    - titel: plot title

    Returns: None

    """
    df.columns = model_names
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    for model_name in model_names:
        path_save = path_save + "_" + model_name
    # path_save = path_save + "_train_" + ".png"
    path_save = path_save + ".png"
    figure, axes = plt.subplots()
    epochs = len(df.index)
    if epochs < 10:
        axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
        axes.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.grid()
    plt.ylabel("Loss")
    plt.xlabel("Total Batches seen")
    plt.title(titel)
    for i in range(df.shape[1]):
        sns.lineplot(data=df, x=df.index + 1, y=df.iloc[:, i], ax=axes, label=df.columns[i])

    plt.legend(loc='upper right')
    # axes.legend(labels=["Acc1", "Acc2"])
    plt.savefig(path_save)
    plt.close()

def compare_models_robustness(train_dataloader, eval_dataloader, test_dataloader, *models: nn.Module, epochs=100 ,noise_levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) -> None:
    from handle_model import handle_model
    from utilities_functions import add_noise_to_mnist_dataset
    acc_list_per_noise_level = []
    # noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for noise_level in noise_levels:
        print(noise_level)
        noisy_train_dataset = add_noise_to_mnist_dataset(train_dataloader.dataset, noise_level=noise_level)
        noisy_train_loader = torch.utils.data.DataLoader(dataset=noisy_train_dataset, batch_size=10, shuffle=False)

        noisy_test_dataset = add_noise_to_mnist_dataset(test_dataloader.dataset, noise_level=noise_level)
        noisy_test_loader = torch.utils.data.DataLoader(dataset=noisy_test_dataset, batch_size=10, shuffle=False)

        noisy_eval_dataset = add_noise_to_mnist_dataset(eval_dataloader.dataset, noise_level=noise_level)
        noisy_eval_loader = torch.utils.data.DataLoader(dataset=noisy_eval_dataset, batch_size=10, shuffle=False)

        model_handlers = []
        model_names = ["Noise Level"]
        for model in models:
            model_handlers.append(handle_model(model, noisy_train_loader, noisy_eval_loader, noisy_test_loader))
            model_names.append(model.__class__.__name__)
            #models[0].__class__.__name__
        acc_list = []
        for model_runner in model_handlers:
            model_runner.run(epochs=epochs, learning_rate=0.001)
            acc_list.append(model_runner.train_acc_with_epoch[-1][-1])

        list_to_append = []
        list_to_append.append(noise_level)
        for acc in acc_list:
            list_to_append.append(acc)

        acc_list_per_noise_level.append(list_to_append)

    import pandas as pd
    df = pd.DataFrame(acc_list_per_noise_level)

    df.columns = model_names
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    figure, axes = plt.subplots()
    axes.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.grid()
    axes.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.ylabel("Accuracy in %")
    plt.xlabel("Noise Level")
    plt.title("Accuracy comparison of NN with MNIST")
        # df = pd.DataFrame(data=acc_list, columns=["Epoch", "Accuracy"])

    for index, row in df.iterrows():
        print(row)
    for i in range(df.shape[1]-1):
        sns.lineplot(data=df, x=df.iloc[:, 0], y=df.iloc[:, i+1], ax=axes, label=df.columns[i+1], marker="*",
                     markersize=8)

    plt.legend(loc='lower right')
        # axes.legend(labels=["Acc1", "Acc2"])
    plt.savefig("Compare_noise_mnist.png")
    plt.close()

    print(acc_list_per_noise_level)


