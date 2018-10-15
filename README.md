# RandomDepthwiseCNN

These codes are used to reproduce the results post on paper 
Image classification and retrieval with random depthwise signed convolutional neural networks.
https://arxiv.org/abs/1806.05789

1. How to reproduce the results of cifar10 part.

	a. Enter folder "cifar10"
	
	b. Run "sh run_job.sh"

	Here is the simple description of reproduce.sh:
		
        python stack_features.py # get 100k features based on Random Depthwise CNN.
        
        python svm.py  # run linearSVC on the new features which just got.

	You will get about 76.36% accuracy on testing dataset in the end.



2. How to reproduce the results of stl10 part.

	a. Enter folder "stl10"

	b. Run "sh run_job.sh"

	Here is the simple description of reproduce.sh:

        python stack_features.py # get 100k features based on Random Depthwise CNN.
        
        python svm.py  # run linearSVC on the new features which just got.
	
	You will get about 71.8% accuracy on testing dataset in the end.



3. How to reproduce the results of sub ImageNet part.

    We choose 10 sub classes from ImageNet to do experiment. Below is the list of folders we chosen.

    ['n03958227', 'n03461385', 'n02814533', 'n02128925', 'n02051845',

    'n03956157', 'n03459775', 'n02808440', 'n02128757', 'n02037110']

    Please create two folders named as "train" and "val" in sub_imagenet directory. Then copy these ten folders from
    ILSVRC2012_img_train to train/ and from ILSVRC2012_img_val to val/ .

	a. Enter folder "sub_imagenet"

	b. Run "sh run_job.sh"

	Here is the simple description of reproduce.sh:

        python stack_features.py # get 100k features based on Random Depthwise CNN.
        
        python svm.py  # run linearSVC on the new features which just got.

    You will get about 78.8% accuracy on testing dataset in the end.


4. How to reproduce the results of MNIST part.

    a. Enter folder "mnist"

	b. Run "sh run_job.sh"

	Here is the simple description of reproduce.sh:

        python stack_features.py # get 100k features based on Random Depthwise CNN.
        
        python svm.py  # run linearSVC on the new features which just got.

    You will get about 99.4% accuracy on testing dataset in the end.


5. How to reproduce the results of cifar100 part.

    a. Enter folder "cifar100"

	b. Run "sh run_job.sh"

	Here is the simple description of reproduce.sh:

        python stack_features.py # get 100k features based on Random Depthwise CNN.
        
        python svm.py  # run linearSVC on the new features which just got.

    You will get about 53.29% accuracy on testing dataset in the end.
    
6. How to reproduce the results of COREAL part.
   
   a. Enter folder "corel"
   
   b. Run "sh run_job.sh"
   
   Here is the simple description of reproduce.sh:
   
        python stack_features.py 
       
        python mlp.py
   You will get features of last second layer of mlp.