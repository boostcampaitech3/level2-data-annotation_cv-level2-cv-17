#!/usr/bin/env bash

for url in $(cat AIHUB_book_paths.txt | tr -d '\r')
do
    mv $url/* /opt/ml/input/data/AIHUB_outside_bookcover/gt
done