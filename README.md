# RandomDepthwiseCNN

These codes are used to reproduce the results post on paper RandomDepthwiseCNN.

1. How to reproduce the results of cifar10 part.

	a. Enter folder "cifar10"
	
	b. Run "sh reproduce.sh"

	Here is the simple description of reproduce.sh:

		python get_data.py # download cifar10's data and save them to current directory.

		sh gen.sh # create ten folders named as features1~features10 to store features.

		python gen_sh.py # generate shell scripts to run python scripts of generate features.
		
		sh generate_features.sh # generate new features and save them into ten folders created previously.
       
		python combine.py # combine all features which saved in those ten folders.

		python sgd_1m.py # Run MultiPerceptron on new features

	You will get about 78.39% accuracy on testing dataset in the end.



2. How to reproduce the results of stl10 part.

	a. Enter folder "stl10"

	b. Run "sh reproduce.sh"

	Here is the simple description of reproduce.sh:

		python get_data.py # download stl10's data and save them to current directory.

		sh gen.sh # create ten folders named as features1~features10 to store features.

		python gen_sh.py # generate shell scripts to run python scripts of generate features.
		
		sh generate_features.sh # generate new features and save them into ten folders created previously.

		python combine.py # combine all features which saved in those ten folders.

		python rh_svc.py # Run linearSVC on new features.
	
	You will get about 70.8% accuracy on testing dataset in the end.



3. How to reproduce the results of sub ImageNet part.

    We choose 10 sub classes from ImageNet to do experiment. Below is the list of folders we chosen.

    ['n03958227', 'n03461385', 'n02814533', 'n02128925', 'n02051845',

    'n03956157', 'n03459775', 'n02808440', 'n02128757', 'n02037110']

    Please create two folders named as "train" and "val" in sub_imagenet directory. Then copy these ten folders from
    ILSVRC2012_img_train to train/ and from ILSVRC2012_img_val to val/ .

	a. Enter folder "sub_imagenet"

	b. Run "sh reproduce.sh"

	Here is the simple description of reproduce.sh:

		python get_data.py # save data to current directory.

		sh gen.sh # create ten folders named as features1~features10 to store features.

		python gen_sh.py # generate shell scripts to run python scripts of generate features.
		
		sh generate_features.sh # generate new features and save them into ten folders created previously.

		python combine.py # combine all features which saved in those ten folders.

		python rh_svc.py # Run linearSVC on new features.

    You will get about 78.4% accuracy on testing dataset in the end.


4. How to reproduce the results of MNIST part.

    a. Enter folder "mnist"

	b. Run "sh reproduce.sh"


	Here is the simple description of reproduce.sh:

		python get_data.py # download MNIST's data and save them to current directory.

		sh gen.sh # create ten folders named as features1~features10 to store features.

		python gen_sh.py # generate shell scripts to run python scripts of generate features.
		
		sh generate_features.sh # generate new features and save them into ten folders created previously.

		python combine.py # combine all features which saved in those ten folders.

		python rh_svc.py # Run linearSVC on new features.

    You will get about 99.4% accuracy on testing dataset in the end.


5. How to reproduce the results of cifar100 part.


    a. Enter folder "cifar100"

	b. Run "sh reproduce.sh"

	Here is the simple description of reproduce.sh:

		python get_data.py # download cifar100's data and save them to current directory.

		sh gen.sh # create ten folders named as features1~features10 to store features.

		python gen_sh.py # generate shell scripts to run python scripts of generate features.
		
		sh generate_features.sh # generate new features and save them into ten folders created previously.

		python combine.py # combine all features which saved in those ten folders.

		python rh_svc.py # Run linearSVC on new features.

    You will get about 53.29% accuracy on testing dataset in the end.
