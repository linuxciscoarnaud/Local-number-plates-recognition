# Local-number-plates-recognition
Local number plates recognition using transfer learning and Deeplearning4j library

This exemple shows how to use transfer learning and Deeplearning4j library to 
implement a License Plate Recogniton system of vehicules from my country (Cameroon). It will perform the recognition of letters and digits that make up license plates.

The final model is transfered from a Tiny YOLO model pretrained on ImageNet and Pascal VOC.
This model is trained on a train dataset of local license plates that I myself created. 

![dataset](https://user-images.githubusercontent.com/1300982/56205630-3ed89480-6042-11e9-9fac-555bf19c95ad.png)

The dataset contains 36 classes with labels distributed as shown on the following figure:

![CLASSES](https://user-images.githubusercontent.com/1300982/56207951-e60bfa80-6047-11e9-99f6-a07f4a14eb7d.png)

Just like the SVHN (Street View House Numbers) dataset, each character (digit or latter) of an image in the dataset is isolated by a character level bounding boxe. The bounding box information are stored in a digitStruct.mat file which can be manipulated with Matlab.
