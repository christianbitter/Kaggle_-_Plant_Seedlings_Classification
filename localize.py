import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PlantSeedlingClassModel import PlantSeedlingClassModel
from common import plot_history, plot_confusion_matrix
from sklearn.metrics import confusion_matrix

# this is a simple localization approach, where we use a sliding window and ask the classifier to give us what it thinks
# since this is a faked image, we have very high accuracy for the small region representing the maize plant
# irrespective of this, we see that outside of the window, we get a strong prediction for fat hen, which is not correct
# ideally, we have a junk class

def class_of(file_name: str):
    return os.path.dirname(file_name).lower()

def main():
    infix    = "small_"
    # infix = ""
    params_fp =  "experiments/base_model/{0}params_classification.json".format(infix)
    final_checkpoint_name = "{0}PlantSeedlingClassModel_final.ckpt".format(infix)
    testing_dir_fp        = "E:/Data/Kaggle_-_Plant_Seedlings_Classification/{0}test".format(infix)
    verbose = True
    m = PlantSeedlingClassModel(params_fp=params_fp, verbose=verbose)
    m.from_checkpoint(checkpoint_name=final_checkpoint_name, full_checkpoint=True)
    img_test_fp = "E:/Data/Kaggle_-_Plant_Seedlings_Classification/fake/maize_fake.png"
    class_name  = "Maize"
    # slide over the image, cut out parts and feed it into predict
    img = Image.open(img_test_fp)
    # plt.imshow(img)
    # plt.show()
    img.convert("RGB")
    target_width, target_height = (100, 100)
    img_resized = img.crop((320, 250, 690, 710))
    img_resized = img_resized.resize((target_width, target_height))
    # plt.imshow(img_resized)
    # plt.show()
    test = m.predict(img_resized)
    print("Predicted: {0} vs. Actual: {1}".format(test["class_name"], class_name))
    if test["class_name"].lower() != class_name.lower():
        raise ValueError("Prediction: {0}".format(test))

    # now let's move over the image
    dim_y = img.height
    dim_x = img.width

    # s_predictions = set()
    for y in range(200, 700, 10):
        for x in range(200, 600, 10):
            px = x
            py = y
            img_predict = img.crop((px, py, px + target_width, py + target_height))
            r = m.predict(img_predict)
            # s_predictions.add(r["class_name"])
            if r["class_name"].lower() == "maize" and r["probability"] > .7:
                print("Predicted ({0}, {1}, {2}, {3}): {4} Prob - Actual: {5}".format(px, py, px + target_width, py + target_height, r["probability"], class_name))
                # plt.imshow(img_predict)
                # plt.show()
                break

if __name__ == '__main__':
    main()