# RandomDepthwiseCNN

These code is used to reproduce the results post on paper RandomDepthwiseCNN.

1. Reproduce the results of cifar10

	a. Enter folder "cifar10"
	b. Run "sh reproduce.sh"

	Here is the simple describtion of reproduce.sh:

		python get_data.py # download cifar10's data and save them to current directory.

		sh gen.sh # create ten folders named as features1~features10 to store features.

		python gen_sh.py # generate shell scripts to run python scripts of generate features.

		python combine.py # combine all features which saved in those ten folders.

		python rh_svc.py # Run LinearSVC on new features
