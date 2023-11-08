import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from cipherTypeDetection.featureCalculations import calculate_histogram, period_ioc_test


def visualize_feature(line, label, index):
    matplotlib.rcParams['interactive'] = False

    features = [x * len(line) for x in period_ioc_test(line)]
    # features = sorted(features)
    # reversed = []
    # for i in range(len(features)):
    #     reversed.append(features[len(features) - 1 - i])
    
    x = range(988)

    plt.ioff()
    fig, ax = plt.subplots()
    ax.scatter(x, features)
    ax.set_title(f"{label}-{index}")
    ax.set_xlabel("Period")
    ax.set_ylabel("Value")

    ax.set(xlim=(0, 988), xticks=np.arange(0, 988),
        ylim=(5, 120), yticks=np.linspace(5, 120, num=20))
    
    fig.savefig(f"/Users/Arbeit/Desktop/visualizations/periodic_hist_test/{label}-{index}.png", format="png")

def visualize_features(lines, label):
    matplotlib.rcParams['interactive'] = False

    x = [[i for i in range(26)] for l in lines]
    y = []
    s = []
    for line in lines:
        feature = [h for h in calculate_histogram(line)]
        # sorted_feature = sorted(feature)
        # reversed = []
        # for i in range(len(sorted_feature)):
        #     reversed.append(sorted_feature[len(sorted_feature) - 1 - i])
        y.append(feature)


    plt.ioff()
    fig, ax = plt.subplots()
    ax.scatter(x, y) # , c=[[line_color] * 4 for line_color in range(len(lines))])
    ax.set_title(f"scatter-{label}")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Value")

    ax.set(xlim=(0, 26), xticks=np.arange(0, 26),
        ylim=(0, 0.1), yticks=np.linspace(0, 0.1, 20))

    fig.savefig(f"/Users/Arbeit/Desktop/visualizations/histograms/scatter-{label}.png", format="png")

def preprocess_ciphertext_line(ciphertext_line):
    def map_text_into_numberspace(text):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        result = []
        for index in range(len(text)):
            try:
                result.append(alphabet.index(text[index]))
            except ValueError:
                raise Exception(f"Ciphertext contains unknown character '{text[index]}'. "
                                f"Known characters are: '{alphabet}'.")
        return result
    cleaned = ciphertext_line.strip().replace(' ', '').replace('\n', '')
    mapped = map_text_into_numberspace(cleaned.lower())
    return mapped
