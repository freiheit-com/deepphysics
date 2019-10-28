# Deep Physics Project

![here should be the title image](https://drive.google.com/uc?export=view&id=18Rav-9Tn1vF79QdQnNVqPMIie87F9ZSN)

<span style="font-size:10px; color:Gray">	&copy; Prof. Dr. Matthias Bode [https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.7b02419](https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.7b02419); this image is not part of the code and therefore NOT licenced under the MIT license!</span>


The project deep physics intends to automatically recognize states of a single molecule utilizing a deep neural network. By comparing the topographic appearance of different states a conventional program was not able to classify large amounts of data sets with a sufficient accuracy. This brings along the time consuming necessity to check all measurements manually. Using an AI, the classification performance can be increased which brings along two main benefits. On the one hand the accuracy of state detection can be increased which leads on the other hand to a reduced time effort for re-examining data making investigations more efficient.
 
## Scientific Background

![here should be the STM scheme image](https://drive.google.com/uc?export=view&id=1maOIoenKDaRCJpgdaQ2Pv5Y1sCYaluiM)

<span style="font-size:10px; color:Gray">	&copy; Finn Christiansen; this image is not part of the code and therefore NOT licenced under the MIT license!</span>


In the research group of Prof. Bode from the University of Würzburg single molecules are utilized to investigate transport properties of different surfaces within short length scales in the order of a few nanometers. To realize these experiments a scanning tunneling microscope (STM) at low temperatures is used where a resolution down to single atoms is possible. In an STM a conductive tip is moved over a conductive surface like an ink printer moves his nozzle over a piece of paper without getting in direct contact. By keeping the tunneling current between tip and sample constant the distance between tip and surface can be used as a measure for the topography. Taking this height information on a defined grid with a certain density of pixel it is possible to generate a topographic height image of a surface. For such an image the tip moves along one line and takes the height information at each pixel. The same is done while the tip moves back the same line before moving down to the next line. 
Such topographies are taken from a single molecule after an excitation by a voltage pulse. For a phthalocyanine the excitation can lead to a tautomerization between four different isomers of the molecule. These isomers can be distinguished by their topographic appearance where the brightest of the four arms marks the state (see 1-4 in the image). For example, picture 1 shows the molecule in state 1 where the brightest arm is on the left side. Two of these state (1 and 2) are stable within measurement times whereas two (3 and 4) are metastable with a lifetime of a few seconds. Consequently, a switching to one of the stable states (1,2) can occur during a scan leading to a discontinuity in the image. This brings along the definition of two intermediate states 5, where the initial state is 3 and final states are 1 or 2, and 6 where the initial state is 4 and the final states are 1 or 2. An example for state 5 is given in image 5 where in the upper part it appears as state 3. After a discontinuity (marked by a dashed line) the molecule switched to state 1. In image 6 an example of state 6 is given where the change is from state 4 to state 2. It is essential to note that states 5 and 6 are artefacts by the scanning method where two states appear as a combined image.
In order to analyze the tautomerization of these single molecules a set of roughly 5000 images is taken for each data point, which easily sums up to 100000 images. For every image the state of the molecule has to be classified which consumes a lot of time for manually recognition. This is where the automatically recognition of deep physics may help for future projects to pre characterize data.




## Using AI for automatic classification of STM images

The problem we are solving is a standard classification problem. Our data set consist of approximately 46 000 manually labeled images of a single molecule in the above mentioned six different states. The metastability of states 3 and 4 brings along an imbalanced number of images for the different states. The majority of the images shows a molecule in states 1 or 2, a minority is in state 3 or 4 and only a few images of states 5 and 6 can be found. 
We solved this task by using a convolutional neural network (CNN) and the new Keras API of Tensorflow 2.0 Beta. In first test runs we got inaccurate results due to the heavily unbalanced data set. Our first approach solving this problem was an augmentation of the data set by generating images for minority states out of majority states. This was error-prone and the accuracy was not satisfying. Additionally, it would have been necessary to adjust the augmentation algorithm for every new experiment geometry. This would have made it impossible to hand the project over to the scientists. Our second and final approach was to use a weighted loss function without an augmentation. This led to satisfying results.  
 
The dataset can be downloaded <span style="font-size:10px; color:Gray"> <a href="https://drive.google.com/uc?export=view&id=10Xs5CfXk2qZVYJDy5TMahj1KvcZz1nfE">here</a> <b>&copy; Prof. Dr. Matthias Bode; the data set is not part of the code and therefore NOT licenced under the MIT license! </b></span>



## Results and Outlook (as of October 2019)

We used an evaluation data set that contained 50 examples per class. With the data set mentioned above, we got an accuracy per class of

<table style="width:50%">
  <tr>
    <th>class</th>
    <th>accuracy</th>
  </tr>
  <tr>
    <td>1</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td>2</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td>3</td>
    <td>0.97</td>
  </tr>
  <tr>
    <td>4</td>
    <td>1</td>
  </tr>
  <tr>
    <td>5</td>
    <td>0.83</td>
  </tr>
  <tr>
    <td>6</td>
    <td>0.86</td>
  </tr>
</table>

The model file that archived this is included in the data set zip file. 

We improved the user experience in multiple iterations to fit the scientist's needs. Now the scientists only need to upload their images to a specific folder of the JupyterLab and execute the predict notebook. The output is a classification file with a state classification for all input images and the certainty of the network for each classification.

The workgroup now has a working solution to solve their task to classify images of a single molecule. Additionally, they re-evaluated their software tool stack and archived a better understanding of their data from a technical point of view by discussing and explaining it to non-scientists.

Multiple members of freiheit.com gained proficiency in deep learning during this project. During this project, the Beta of Tensorflow 2.0 including the new high-level API of Keras was released. In this project, we tried out the new possibilities. The high-level API accelerates working with neural networks because many common functions are abstracted. During this project, Daniel Bartz proposed the usage of Google's hosted solution for JupyterLabs. We liked it and switched completely over to it in this project. We always shared our new experiences and learnings in the company internal _Data Science Faction_. This helped us in our customer projects, too.

The research group is currently not investigating systems where the here presented task of image classification fits. For future projects it is very likely that the notebooks for classifications can be utilized to perform a faster data analyzation. Moreover, different projects bring along the necessity to classify objects on images which is beyond the scope of this project. Nevertheless, the results of this projects and the knowledge gained within the process of establishing this classification tool could be taken as a base to establish the object classification for further application in the context of a master's thesis.


  
_**Disclaimer**_: The AI will only be used to pick promising measurement series out of the pool of measurement series’. Scientists will always double-check all images that are used for scientific publications.


## freiheit.com and the research group are hiring!

You are seeking for new challenges in Software Development or Solid-State Physics? [freiheit.com](https://hire.withgoogle.com/public/jobs/freiheitcom/view/P_AAAAABkAAAoDBd5NKppMHl?trackingTag=gitHub) and the [research group of Prof. Dr. Bode](https://www.physik.uni-wuerzburg.de/ep2/home/) are hiring and are happy to receive your application documents!

## Contribution

The project’s source code is free software - free as in freedom and not as in free beer (MIT license). Contributions are welcome and we are open to questions and new exciting projects. Feel free to contact us at finn.christiansen@freiheit.com.

## Credits

This project is a collaboration between the physics research group of Prof. Dr. Bode (Head of Chair, Professor at Experimental Physics 2, University of Würzburg) and freiheit.com technologies gmbh (founded by Stefan Richter and Claudia Dietze)).

### freiheit.com

- [Finn Christiansen](https://www.xing.com/profile/Finn_Christiansen3/) has initiated the project, is the communication leader and a core contributor.
- [Vladimir Kravtsov](https://github.com/vladhc) has implemented the initial proof of concept and is a core contributor.
- [Rasmus Buchmann](github.com/rbuchmann) supports the project as a advisor.
- [Stefan Richter](https://www.linkedin.com/in/smartrevolution/) is the founder and CTO of freiheit.com. He is enthusiastic about the possibilities of AI, pushing AI projects  and provides company time as well as computing resources.

### Research Group Bode

- [Prof. Dr. Matthias Bode](https://www.physik.uni-wuerzburg.de/ep2/team/prof-dr-matthias-bode/) has initiated the project, is the scientific leader and provider of the classified data.
- [Dr. Jens Kügel](https://www.physik.uni-wuerzburg.de/ep2/team/dr-jens-kuege) was supervising the project as group leader and STM expert
- [Markus Leisegang](https://www.physik.uni-wuerzburg.de/ep2/team/markus-leisegang/) is responsible for communication, providing, testing and applying data to the neural network

### Extern

- [Dr. Daniel Bartz](https://www.linkedin.com/in/bartzdaniel/) supports the project by reviewing code & analyses, refactoring and providing advice.
