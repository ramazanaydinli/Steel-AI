# Steel-AI
 AI applications for steel structures problems

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
 ![image](https://user-images.githubusercontent.com/80748060/193355352-200654ab-595d-44dd-b3df-fa6f616a4b99.png)

