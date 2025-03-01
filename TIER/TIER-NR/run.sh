#!/bin/bash

for benchmark in  AGIQA1K AGIQA3K AIGCIQA2023q AIGCIQA2023a AIGCIQA2023c 
do  
    echo "python -u train.py  --log_info=$benchmark   --benchmark=$benchmark "
    python -u train.py  --log_info=$benchmark   --benchmark=$benchmark 
done