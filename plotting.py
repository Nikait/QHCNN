import matplotlib.pyplot as plt
import seaborn as sns


def plot_history(path: str, loss: list[float], score: list[float]):
    fig, axes = plt.subplots(1, 2, figsize=(30, 7), sharey=True)
    sns.set_style("whitegrid")
    k = list(range(1, len(loss)+1))


    axes[0].set(xlabel='epochs', ylabel='loss')
    axes[1].set(xlabel='epochs', ylabel='test accuracy')
    sns.lineplot(x=k, y=loss, ax=axes[0])
    sns.lineplot(x=k, y=score, ax=axes[1])

    plt.savefig(path)
