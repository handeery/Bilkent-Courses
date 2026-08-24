import matplotlib.pyplot as plt
from IPython import display

plt.ion()

# Dictionary to store figures for each plot type
figures = {'score': None, 'food': None, 'poisonous': None}

def plot(scores, mean_scores, plot_type='score'):
    if plot_type not in figures:
        raise ValueError("Invalid plot_type. Use 'score', 'food', or 'poisonous'.")

    fig = figures[plot_type]

    if fig is None:
        fig, ax = plt.subplots()
        figures[plot_type] = fig
    else:
        plt.figure(fig.number)  # Switch to the existing figure

    display.clear_output(wait=True)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')

    if plot_type == 'score':
        plt.ylabel('Score')
    elif plot_type == 'food':
        plt.ylabel('Food eaten')
    elif plot_type == 'poisonous':
        plt.ylabel('Poisonous Food eaten')

    plt.plot(scores, label='Scores')
    plt.plot(mean_scores, label='Mean Scores')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)
