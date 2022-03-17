# Problem

Successful cultivation of maize depends largely on the efficacy of weed control. Weed control during the first six to eight weeks after planting is crucial, because weeds compete vigorously with the crop for nutrients and water during this period. Annual yield losses occur as a result of weed infestations in cultivated crops. Crop yield losses that are attributable to weeds vary with type of weed, type of crop, and the environmental conditions involved. Generally, depending on the level of weed control practiced yield losses can vary from 10 to 100 %. Rarely does one experience zero yield loss due to weeds... Yield losses occur as a result of weed interference with the crop's growth and development....This explains why effective weed control is imperative. In order to do effective control the first critical requirement is correct weed identification.

## Model

### [MobileNet V2](https://keras.io/api/applications/mobilenet/#mobilenetv2-function)

MobileNet is a class of efficient Convolutional Neural Networks for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks

![MobileNet Architecture](https://miro.medium.com/max/1384/1*7R068tzqqK-1edu4hbAVZQ.png "MobileNet Architecture")

For our use case we have updated the hidden fully connected (Dense) layer for a single layer of 1024 neurons to 3 layers of sizes 128, 256 and 128 neurons respectively followed by the output layer with 12 neuron giving the probability for each of the 12 classes.

## Dataset

### [V2 Plant Seedling Dataset ](https://www.kaggle.com/vbookshelf/v2-plant-seedlings-dataset)

This dataset contains 5,539 images of crop and weed seedlings. The images are grouped into 12 classes as shown in the above pictures. These classes represent common plant species in Danish agriculture. Each class contains rgb images that show plants at different growth stages. The images are in various sizes and are in png format.
