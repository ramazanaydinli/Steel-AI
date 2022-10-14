# Steel-AI
 AI applications for steel structures problems, typical problem which will be feeded into system as an input is shown below:
![image](https://user-images.githubusercontent.com/80748060/195896003-9210d54f-f406-45d6-a5e2-d4e115f6e306.png)
Problems are surely including text but it may also include technical drawings

Solution Idea:
Due to irregular structure of  input, it is not possible to correctly extract all the features at the expected order since the manpower and investment is not enough. But %70 accuracy could be reached and it is enough.
- Field Research
After some investigation, related article which is published 6 months ago is obtained. Link of the article: https://arxiv.org/ftp/arxiv/papers/2205/2205.02659.pdf
From the paper it can be seen that it is possible to reach %80 accuracy for both shape detection and dimension extraction. Methodology of the paper will be followed.

- Prototype Work
Firsty, input will be feeded into pre-trained neural network to obtain boundaries of text and technical drawings. Expected output is below:
![image](https://user-images.githubusercontent.com/80748060/195900272-a7040dc0-1da9-4651-9244-26f427b5f208.png)

After that text part will be used for NLP and drawing parts will be used for detailed processing using OpenCV. Obtaining information from text is relatively easy, so first focus point will be on drawings.
For this purpose a neural network should be used but it is not possible to obtain enough data to train on. Article above suggests creating this data manually. At first four steps this manually created data is used but there are some misclassifications done by model. After some more research it is discovered that Generative Adverserial Networks (GANs) can be used to create artificial data to overcome this problem. But this approach only solves style of the drawings, also dimensions should be obtained. At the end of 1 year of this project, this much is done. For now, work is focused on tables that are containing section data.

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

