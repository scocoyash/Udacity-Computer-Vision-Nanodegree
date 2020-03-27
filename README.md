## Udacity-Computer-Vision-Nanodegree
----
    This repository contains code exercises, my solutions and materials for Udacity's Computer Vision Nanodegree program. 
    It consists of tutorial notebooks that demonstrate, or challenge you to complete, various computer vision applications and techniques.

### Steps to run this repository :

We are going to use **Anaconda** command prompt for the whole exercise.
These instructions also assume you have `git` installed for working with Github from a terminal window.

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
    ```
    git clone https://github.com/scocoyash/Udacity-Computer-Vision-Nanodegree.git
    cd Udacity-Computer-Vision-Nanodegree/
    ```

2. Create (and activate) a new environment, named `` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__: 
	```
	conda create -n cvnd python=3.6
	source activate cvnd
	```
	
	 The `(cvnd)` at the start of terminal indicates that your environment has been activated, and you can proceed with further package installations.

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__: 
	```
	conda install pytorch torchvision -c pytorch 
	```

4. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
    ```
    pip install -r requirements.txt
    ```

7. That's it! Enjoy :)
