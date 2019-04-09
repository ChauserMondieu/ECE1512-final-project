import matplotlib.pyplot as plt


def plot_each(title, x_label, y_label, x_data, y_data, save_path):
    plt.title( title )
    plt.xlabel( x_label )
    plt.ylabel( y_label )
    plt.plot( x_data, y_data, color='green', label='training set' )
    # plt.legend( loc='upper right' )
    plt.savefig(save_path, dpi = 600)
    plt.cla()
