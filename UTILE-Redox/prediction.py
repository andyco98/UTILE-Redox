import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import glob
import cv2
import os
import numpy as np
from PIL import Image
from keras.models import load_model



def prediction(image_path, model_path, mask_folder):
    #Resizing images, if needed
    SIZE_X = 512
    SIZE_Y = 512
    n_classes=4 #Number of classes for segmentation
    path = model_path


    ### FOR NOW LET US FOCUS ON A SINGLE MODEL
    val_images = []

    directory_path = image_path
    img_paths = glob.glob(os.path.join(directory_path, "*.tif")) + glob.glob(os.path.join(directory_path, "*.tiff"))+ glob.glob(os.path.join(directory_path, "*.png"))+ glob.glob(os.path.join(directory_path, "*.jpg"))+ glob.glob(os.path.join(directory_path, "*.jpeg"))
    # Normalize and correct path
    img_paths = [os.path.normpath(path) for path in img_paths]   
    # Sort image paths numerically
    img_paths.sort(key=lambda fname: int(os.path.splitext(os.path.basename(fname))[0][6:]))
    #print(img_paths)

    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Failed to read the image: {img_path}")
            continue
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (SIZE_X, SIZE_Y))
        val_images.append(img)

    val_images = np.array(val_images, dtype=np.float32) / 255.0
    #Convert list to array for machine learning processing        

    model1 = load_model(path, compile = False)

    for idx, img in enumerate(val_images):
        test_img_input=np.expand_dims(img, 0)
        print(test_img_input.shape)
        test_pred1 = model1.predict(test_img_input)
        test_prediction1 = np.argmax(test_pred1, axis=-1)[0,:,:]
        test_prediction1 = test_prediction1.astype(np.uint8)
        im = Image.fromarray(test_prediction1)
        im.save(os.path.join(mask_folder, f"{idx}_pred.tif"))
        if idx>=1:
            continue
    return
