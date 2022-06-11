import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    name = './FederatedLoss/FL'
    name_txt = name + '.txt' #I can switch out the file here that I want to generate an image for

    loss_data = np.loadtxt(name_txt)
    epoch_num = range(0, len(loss_data))

    fig = plt.figure(figsize=(10, 5))
    plt.plot(epoch_num, loss_data, label="FL-loss.py")

    plt.xlabel('Epochs')
    plt.ylabel('Loss - test set')
    plt.title(name)
    plt.legend()

    fig.savefig( name + '.jpg', bbox_inches='tight', dpi=150)



