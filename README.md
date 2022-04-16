# saturdaysAI_dl_mad22

Content and practices from course done at [Saturdays.AI](!https://saturdays.ai/) **5th Ed** in Madrid for the **Deep Learning** path course.

The course is structured with content on their [Eduflow platform](!saturdays.eduflow.com) to review before each class and practices done in our own and then peer-reviewed through the platform

For each week there is an **individual practice** available and a **group challenge** that is done after the weekly class; not all practices were done.

Al the models done and the notebooks have been developed in google colab, making use of their free CPU.

---

### Contents:

* [**week 01 - Classification and Main Components**](!)  

  

  * [Prediction of types of ECG for PhysionNet's data](!https://github.com/LAMDAMielgo/saturdaysAI_dl_mad22/blob/1103300ed39c62575b858e6a6100841f72db7318/week_1/practice/DL_S1_Practice_ECG.ipynb)

    Using a Sequential CNN based on a paper, reached a 95% accuracy, with an accuracy of 0.78% on arrhythmia cases (paper goal was 93%).

  * [Prediction of categorical proximity to ocean for California Real Estate data](!https://github.com/LAMDAMielgo/saturdaysAI_dl_mad22/blob/1103300ed39c62575b858e6a6100841f72db7318/week_1/challenge/DL_S1_Challenge_Oceans.ipynb)

    Using Kaggle's dataset for california house prices, the notebooks explores the prediction of the houses' ocean proximity.

    Used geospatial techniques for feature engineering in order to create an approximation of urban density.

    Reached a 93% accuracy on label **near_ocean**.

    

* [**week 02 - Convolutional Neural Networks**](!)

  * **[ #todo ]** Image classifier cats vs dogs

  * [Image classifier Gandalf vs Dumbledore.](!https://github.com/LAMDAMielgo/saturdaysAI_dl_mad22/blob/main/week_2/challenge/DL_S2_CNN_Challenge_GvD.ipynb)

    Using data augmentation and EfficientNetB0 for transfer learning.

    Trained with only 40 images, so the goal was to train a good enough model, being very sensible to overfitting.

    

* [**week 03 - Object Detection and Segmentation**](!)  

  * **[# no practice]**

  * Head pose object detection

    The objective of the practice is to detect to which direction a person is looking (head orientation) using [this](!http://crowley-coutaz.fr/Head%20Pose%20Image%20Database.html) database.

* [**week 04 - Autoencoders and intro to Generative DL**](!)  

  * VAE for (sic?)
  * Compression of Youtube video to create new video frames
  * 

* [**week 05 - Reinforcement Learning**](!)  

  * Actor-Critic for 
  * 

* [**week 06 - DL for NLP**](!)

---

#### Credits

The content of this repository is available through Saturdays.ai platform, the base notebooks, the data and the resources were provided mostly by them.

The challenges where done in groups of three-four people, though in here I have done some tweaks to the groups assignments.

[#todo] credit everybody from haifflepuff.

---

#### Repository Structure

Each week is divided divided following this structure:

```

 /week_[xx]/
   ├── DL_S[x]_notes.md
   ├── practice/
   | 	 ├── DL_S[x]_practice_[name].ipynb
   |     ├── .data/
   |     ├── module/   							-- sometimes code is saved in a module
   |     └── model/
   |		  └── best_model.h5
   |
   └── challenge/
    	├── DL_S[x]_challenge_[name].ipynb
        ├── .data/
        └── model/
    		  └── best_model.h5
```
