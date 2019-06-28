# Deep Physics Project

![here should be the title image](https://drive.google.com/uc?export=view&id=18Rav-9Tn1vF79QdQnNVqPMIie87F9ZSN)

<span style="font-size:10px; color:Gray">	&copy; Prof. Dr. Matthias Bode [https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.7b02419](https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.7b02419); this image is not part of the code and therefore NOT licenced under the MIT license!</span>

This project is an automation of molecule state recognition using a deep neural network. For the specific use case, a conventional software program was not able to classify the molecule states reliably enough. Thus, without an AI, it is only possible to classify the data manually. The main benefit of using the AI is a dramatically increased classification performance.

## Scientific Background

![here should be the STM scheme image](https://drive.google.com/uc?export=view&id=1maOIoenKDaRCJpgdaQ2Pv5Y1sCYaluiM)

<span style="font-size:10px; color:Gray">	&copy; Finn Christiansen; this image is not part of the code and therefore NOT licenced under the MIT license!</span>

The research group of Prof. Bode from Würzburg analyzes the states of a single molecule after excitation to investigate transport properties of different materials. For this purpose, they are using a scanning tunneling microscope (STM).  
The STM moves a tip over a specimen like an ink printer moves his nozzle over a piece of paper. That is why it is called “scanning”. Between specimen and tip, a constant voltage is applied. For each position, the current that flows through the system is held constant by regulating the tip height. Therefore, a scan image is essentially the topographic height map of a sample. Six representative images are shown below. Because of the scanning movement of the tip, the images are shown one row after another while scanning. The tip moves full right, full left, one row down, full left and so on.  
After the molecule is excited, it is in one of four states (see 1-4 in the image). The states can be distinguished by finding the brightest “arm”. In picture 1, the brightest arm is the left one, which defines the state 1.  
Two of these states are metastable (3,4) with a lifetime of a few seconds. Consequently, a switching to one of the stable states (1,2) can occur during a scan. If this happens, there is a discontinuity in the image. Images 5 and 6 are examples of that. The upper part of image 5 looks like state 3. Then there is the discontinuity (marked by the dashed line). After that, the image looks like state 1, which is defined as state 5. Image 6 shows state 6 (first 4 then 2). It is essential that in reality, the molecule is never in state 5 or 6. These states are only artifacts of the nature of the scanning method.

## Using AI for automatic classification of STM images

The problem we are solving is a standard classification problem. We have a data set of approximately 10k manually labeled images of molecules. Dataset is unbalanced, so we used image augmentation for creating more images of states 5 and 6. 

## Roadmap

The goal of this project is to automate the tedious process of image classification. The AI might be better as a human, but this a pleasant bonus. Our goal for the end of August is that the average accuracy is at least 95%. A scientist then double-checks the images for which the network calculated a low confidence value.  
_Disclaimer_: The AI will only be used to pick promising measurement series out of the pool of measurement series’. Scientists will always double-check all images that are used for scientific publications.

## Setup

We are using [JupyterLab](https://jupyter.org/) to develop, train and apply the neural network. One can either use JupyterLabs hosted by [Google’s AI Platform](https://cloud.google.com/ml-engine/docs/notebooks/overview) or by a local Docker instance.

## freiheit.com and the research group are hiring!

You are seeking for new challenges in Software Development or Solid-State Physics? [freiheit.com](https://hire.withgoogle.com/public/jobs/freiheitcom/view/P_AAAAABkAAAoDBd5NKppMHl?trackingTag=gitHub) and the [research group of Prof. Dr. Bode](https://www.physik.uni-wuerzburg.de/ep2/home/) are hiring and are happy to receive your application documents!

## Contribution

The project’s source code is free software - free as in freedom and not as in free beer (MIT license). Contributions are welcome and we are open to questions and new exciting projects. Feel free to contact us at finn.christiansen@freiheit.com.

## Credits

This project is a collaboration between the physics research group of Prof. Dr. Bode (Head of Chair, Professor at Experimental Physics 2, University of Würzburg) and freiheit.com technologies gmbh (founded by Stefan Richter and Claudia Dietze)).

### freiheit.com

- [Finn Christiansen](https://www.xing.com/profile/Finn_Christiansen3/) has initiated the project, is the communication leader and a core contributor.
- [Vladimir Kravtsov](https://github.com/vladhc) has implemented the initial proof of concept and is a core contributor.
- [Dr. Daniel Bartz](https://www.linkedin.com/in/bartzdaniel/) supports the project by reviewing code & analyses, refactoring and providing advice.
- [Rasmus Buchmann](github.com/rbuchmann) supports the project as a advisor.
- [Stefan Richter](https://www.linkedin.com/in/smartrevolution/) is the founder and CTO of freiheit.com. He is enthusiastic about the possibilities of AI, pushing AI projects  and provides company time as well as computing resources.

### Research Group Bode

- [Prof. Dr. Matthias Bode](https://www.physik.uni-wuerzburg.de/ep2/team/prof-dr-matthias-bode/) has initiated the project, is the scientific leader and provider of the classified data.
- [Dr. Jens Kügel](https://www.physik.uni-wuerzburg.de/ep2/team/dr-jens-kuege) and [Markus Leisegang](https://www.physik.uni-wuerzburg.de/ep2/team/markus-leisegang/) testing and applying the neural network and are domain experts.