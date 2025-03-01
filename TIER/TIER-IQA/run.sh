#!/bin/bash
for benchmark in  AGIQA1K AGIQA3K AIGCIQA2023q AIGCIQA2023a AIGCIQA2023c 
do  
    echo "python -u train.py --lr=1e-5 --model=StairIQA --log_info=$benchmark   --benchmark=$benchmark "
    python -u train.py --lr=1e-5 --model=StairIQA --log_info=$benchmark   --benchmark=$benchmark 
done

for benchmark in  AGIQA1K AGIQA3K AIGCIQA2023q AIGCIQA2023a AIGCIQA2023c 
do  
    echo "python -u train.py --lr=1e-4 --model=HyperIQA --log_info=$benchmark   --benchmark=$benchmark "
    python -u train.py --lr=1e-4 --model=HyperIQA --log_info=$benchmark   --benchmark=$benchmark 
done

for benchmark in  AGIQA1K AGIQA3K AIGCIQA2023q AIGCIQA2023a AIGCIQA2023c 
do  
    echo "python -u train.py --lr=1e-5 --model=LinearityIQA --log_info=$benchmark   --benchmark=$benchmark "
    python -u train.py --lr=1e-5 --model=LinearityIQA --log_info=$benchmark   --benchmark=$benchmark 
done


