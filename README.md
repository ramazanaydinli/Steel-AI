# Steel-AI


 Lvl-0
 - Starting level of the project
 - No order, just the backbone created
 
 Lvl-1
 - Two types of generators creating section images and just white blank images
 - Detector takes these two type and trained on them
 ![image](https://user-images.githubusercontent.com/80748060/193341808-57f6309e-774e-4635-aa0c-1c533b789d7f.png)



 
 Lvl-2
 - Drawing type updated, now shapes could be hatched with white, gray or black
 - Detector has validation set, input shape changed
 ![image](https://user-images.githubusercontent.com/80748060/193342306-cac40d9f-0dd2-4444-8aa3-7fbfb608fc77.png)



 
 
 
 Lvl-3
 - Drawings now contain annotations
 - Detector now making multi-classification instead of binary
 - Detector now has dropout layers and data augmentation, images turned into grayscale before training
 ![image](https://user-images.githubusercontent.com/80748060/193342952-7ddb6787-433a-4064-8a9e-91f33946c2f6.png)

 
 

Lvl-4
 - InceptionV3 model implemented
 - Since dataset created manually it is not possible to add more features which causing poor prediction performance, misclassified image is below
 - For overcoming this problem GANs will be used for creating dataset
 
![image](https://user-images.githubusercontent.com/80748060/193355486-2a883f0d-8aba-441f-9032-f7a16fe36a53.png)

Lvl-5
 - Neural Style Transfer method used and misclassification problem solved
 - It is also important to realize, dimensional differences creating more problem comparing with the style differences
 ![image](https://user-images.githubusercontent.com/80748060/194715308-389e6a86-3d78-4004-8575-e8266c5ed145.png)
 
 
 Since the dataset created manually by random numbers, it is not a bad idea to create them from industry-standardized dimensions. Hard part of this work is completed and now tables can be detected by using keras-ocr.
 
 
 ![image](https://user-images.githubusercontent.com/80748060/197308936-01d3a045-c170-4982-b54d-66ff1f7cc910.png)


Lvl-6
 - Multi-label object detection environment created. Inference from Resnet-50 model done. Now model can recognize and localize section from any input image.
 - These two images below is similar to what model trained on and it is working perfectly.
 
 ![image](https://user-images.githubusercontent.com/80748060/206924613-1095ca88-767b-45bb-8ec8-bc9f9a94a8ad.png)


- Model never see these three images below but it recognizes them as well.

![image](https://user-images.githubusercontent.com/80748060/206924660-0e4a2473-878b-48de-a75f-8bdd2241a547.png)


 

