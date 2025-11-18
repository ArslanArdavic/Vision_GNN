#!/usr/bin/env bash
echo "Restructuring validation ..."
cd imageNet_sub/validation

# Python like zip from two streams
function zip34() { while read word3 <&3; do read word4 <&4 ; echo $word3 $word4 ; done }
#curl -L -o imagenet_2012_validation_synset_labels.txt  https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt
find . -name "*.JPEG" | sort > images.txt
zip34 3<images.txt 4<imagenet_2012_validation_synset_labels.txt | xargs -n2 -P8 bash -c 'mkdir -p $2; mv $1 $2' argv0

#rm *.txt
cd ..

echo "val:" $(find imageNet_sub/validation -name "*.JPEG" | wc -l) "images"