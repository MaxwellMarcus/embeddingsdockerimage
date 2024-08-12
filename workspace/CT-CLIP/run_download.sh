#! /bin/bash


N=0

while [ $N == 0 ];
do
	echo "Starting with folder `ls batch -v | head -1`"
	python scripts/xnat_zero_shot.py `ls batch -v | head -1`
	N = $?
done

