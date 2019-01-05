import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PlantSeedlingClassModel import PlantSeedlingClassModel
from common import plot_history, plot_confusion_matrix
from sklearn.metrics import confusion_matrix

def class_of(file_name: str):
    return os.path.dirname(file_name).lower()

def main():
    infix    = "small_"
    # infix = ""
    params_fp =  "experiments/base_model/{0}params_classification.json".format(infix)
    final_checkpoint_name = "{0}PlantSeedlingClassModel_final.ckpt".format(infix)
    testing_dir_fp        = "E:/Data/Kaggle_-_Plant_Seedlings_Classification/{0}test".format(infix)
    train_fresh = True
    eval = False
    verbose = True
    plot_history = False
    m = PlantSeedlingClassModel(params_fp=params_fp, verbose=verbose)

    if train_fresh:
        m.import_data()
        m.build()
        m.prepare()
        if verbose:
            print(m.summary())

        training_history = m.train()
        m.to_checkpoint(final_checkpoint_name)
        if plot_history:
            plot_history(training_history)
    else:
        m.from_checkpoint(checkpoint_name=final_checkpoint_name, full_checkpoint=True)

    if eval:
        eval_result = m.evaluate()
        print("Evaluation Results ({0}): {1}".format(m.model.metrics_names, eval_result))

    x = m.predict_from_directory(input_dir_fp=testing_dir_fp)
    actual_class = [class_of(fn) for fn in x['file_names']]
    predicted_class = [s.lower() for s in x['predicted_class_name']]
    print("y vs. yhat ({0} vs. {1})".format(len(actual_class), len(predicted_class)))
    # print("Y:\r\n{0}".format(actual_class))
    # print("Yhat:\r\n{0}".format(predicted_class))
    cm = confusion_matrix(actual_class, predicted_class, m.classes())
    plot_confusion_matrix(cm, classes=m.classes(), normalize=True)
    plt.show()


if __name__ == '__main__':
    main()