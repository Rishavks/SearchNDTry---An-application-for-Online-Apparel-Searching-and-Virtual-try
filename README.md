# SearchNDTry---An-application-for-Online-Apparel-Searching-and-Virtual-try
SearchNDtry is an image based search and try-on application with web and mobile interface both; wherein the user can look-up visually similar apparels online and try-on apparels virtually.  


![image](https://user-images.githubusercontent.com/55141040/156546219-5792651a-967b-4a7d-87c3-a513986cc4a7.png)

Sign-In Page


![image](https://user-images.githubusercontent.com/55141040/156546370-aa2757bd-d509-4c74-bad4-22d3e4dc8379.png)

Welcome Page


![image](https://user-images.githubusercontent.com/55141040/156546478-9cf07189-dfe8-48e2-a39b-1b953afbf971.png)

Dark Mode Enabled


![image](https://user-images.githubusercontent.com/55141040/156546575-64eece41-cf20-45c4-b17c-462374c2f15d.png)

Video Tutorials for using our website


![image](https://user-images.githubusercontent.com/55141040/156546713-f5832a46-697f-42d6-8c2b-9ad460e79df5.png)

Searching an image through file upload


![image](https://user-images.githubusercontent.com/55141040/156546895-ab32f71f-05b7-47dd-9871-e06d1ff8020a.png)

Searching Image via Webcam



The search feature is implemented with the help of Google Vision Api, which takes an input query image from the user and returns a list of visually similar images available on online e-commerce websites.


![image](https://user-images.githubusercontent.com/55141040/156547037-cc2fcb9f-cb91-45f7-a2fd-bc8e4f098a01.png)

Searched apparels can be added to cart


A cart feature is implemented using Google Firebase's Realtime Database, which enables the user to save apparel images in their respective carts for further access.

![image](https://user-images.githubusercontent.com/55141040/156547139-f220d3fe-f441-4541-a25c-196fac77180b.png)

User Cart containing apparels


After choosing an apparel to try-on virtually, the user needs to upload his/her photo, our app allows them to upload an image file or capture one via webcam.


The virtual try feature is implemented using deep learning models and image processing, which extracts the 2D cloth from a selected image and wraps it on the user image to provide a preview before purchasing.


On selection of an image, the apparel will be segmented from the background image using Fashion-AI-Segmentation and PixelLib library, and wrapped on the segmented user image using a deep learning model, which is a multi-stage virtual try-on deep neural network based on JPP-Net and CP-VTON.


![image](https://user-images.githubusercontent.com/55141040/156547349-eb4997b7-6600-4cd1-b949-846749c709ea.png)

Apparel Image from Cart


![image](https://user-images.githubusercontent.com/55141040/156547398-84355d47-18c4-47d0-b525-185a066d51b0.png)

Extracting Cloth from the image(processing using pixellib)


![image](https://user-images.githubusercontent.com/55141040/156547512-82b8c46a-79a8-4892-afb6-52b5664b597f.png)

Extracting user from the image


![image](https://user-images.githubusercontent.com/55141040/156547587-4ce8aa29-da6f-48b4-8487-59fa126c2afc.png)

Wrapping cloth on user

References:

[1] https://cloud.google.com/vision

[2] https://pixellib.readthedocs.io/en/latest/ 

[3] Han, X., Wu, Z., Wu, Z., Yu, R., & Davis, L.S. (2018). VITON: An Image-Based Virtual Try-on Network. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7543-7552.

[4] https://getbootstrap.com/docs/5.1/getting-started/contents/

[5] https://firebase.google.com/docs



