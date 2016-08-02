from tkinter.filedialog import askopenfilename, askdirectory, Tk
from matplotlib import pyplot as plt
from matplotlib import rc

def ask_image_path():
    root = Tk()
    img_path = askopenfilename(filetypes = (("jpeg files","*.jpg"),
                                            ("all files","*.*")))
    root.withdraw()
    root.destroy()
    return img_path
    
def ask_dir():
    root = Tk()
    img_path = askdirectory()
    root.withdraw()
    root.destroy()
    return img_path
    
def show_hist(hist, bins, xticks=None):
    width = 1.0 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    rc('font', family='DejaVu Sans')
    if xticks:
        plt.xticks(range(len(xticks)), xticks)
    plt.bar(center, hist, align='center', width=width)
    plt.show()