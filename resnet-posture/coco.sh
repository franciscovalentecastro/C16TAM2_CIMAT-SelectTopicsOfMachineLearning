# mkdir coco
# cd coco
# mkdir images
# cd images
cd coco/images

# wget http://images.cocodataset.org/zips/train2017.zip
# wget http://images.cocodataset.org/zips/val2017.zip
# wget http://images.cocodataset.org/zips/test2017.zip
# wget http://images.cocodataset.org/zips/unlabeled2017.zip

unzip train2017.zip | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'
unzip val2017.zip | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'
unzip test2017.zip | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'
# unzip unlabeled2017.zip | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'

rm train2017.zip
rm val2017.zip
rm test2017.zip
rm unlabeled2017.zip 

cd ../
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

unzip annotations_trainval2017.zip | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'
unzip stuff_annotations_trainval2017.zip | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'
unzip image_info_test2017.zip | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'
unzip image_info_unlabeled2017.zip | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'

rm annotations_trainval2017.zip
rm stuff_annotations_trainval2017.zip
rm image_info_test2017.zip
rm image_info_unlabeled2017.zip