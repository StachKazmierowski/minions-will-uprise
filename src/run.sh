#!/bin/bash

for resource in {15..30}
do
	python3 test_mwu.py $resource 15 7 13 >> tmp$resource15 &
done
