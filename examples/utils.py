import os


def get_plot_path(filename: str):
    folder = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(folder, "plots"), exist_ok=True)
    return os.path.join(folder, "plots", filename)
