#!/bin/sh


python get_data.py

sh gen.sh

python gen_sh.py

sh generate_features.sh

python combine.py

python rh_svc.py


