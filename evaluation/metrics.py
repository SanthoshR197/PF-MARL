import matplotlib.pyplot as plt
import numpy as np

def plot_infection_curve(infection_history):
    plt.figure()
    plt.plot(infection_history)
    plt.xlabel("Days")
    plt.ylabel("Total Infection")
    plt.title("Epidemic Infection Curve")
    plt.show()