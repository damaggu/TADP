initial_dir=$(pwd)

# Perform actions based on the argument
if [ "$1" = "seg" ]; then
  echo "Downloading ADE20k"

cd data &&
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip && unzip ADEChallengeData2016.zip && rm ADEChallengeData2016.zip &&
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016/annotations.zip && unzip annotations.zip && rm annotations.zip &&
wget http://data.csail.mit.edu/places/ADEchallenge/release_test.zip && unzip release_test.zip && rm release_test.zip &&
cd ..
fi

if [ "$1" = "depth" ]; then
echo "Downloading nyu"
mkdir data/nyu_depth_v2 &&
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat -O data/nyu_depth_v2/nyu_depth_v2_labeled.mat  &&
python datasets/depth/nyu_extract_official_train_test_set_from_mat.py data/nyu_depth_v2/nyu_depth_v2_labeled.mat datasets/depth/nyu_splits.mat data/nyu_depth_v2/official_splits/ &&
cd data/nyu_depth_v2 && mkdir sync && cd sync &&
gdown https://drive.google.com/uc?id=16JqgvqtICEsyi6dX85F6BB7AwI0qF9aP &&
unzip sync.zip && rm sync.zip
fi


if [ "$1" = "cross" ]; then
cd data
mkdir dark_zurich && cd dark_zurich
wget https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_val_anon.zip && unzip Dark_Zurich_val_anon.zip && rm Dark_Zurich_val_anon.zip
cd ../
mkdir NighttimeDrivingTest && cd NighttimeDrivingTest
wget http://data.vision.ee.ethz.ch/daid/NighttimeDriving/NighttimeDrivingTest.zip && unzip NighttimeDrivingTest.zip && rm NighttimeDrivingTest.zip
cd ../../

# download and put cityscapes zips in /data/cityscapes/
mv ./data/cityscapes/gtFine_trainvaltest/gtFine ./data/cityscapes/
mv ./data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit ./data/cityscapes/
mv ./data/cityscapes/leftImg8bit_trainextra/leftImg8bit ./data/cityscapes/
rm -rf ./data/cityscapes/gtFine_trainvaltest
rm -rf ./data/cityscapes/leftImg8bit_trainvaltest
rm -rf ./data/cityscapes/leftImg8bit_trainextra
git clone https://github.com/open-mmlab/mmsegmentation.git
python -m pip install cityscapesscripts
mim install mmengine
python mmsegmentation/tools/dataset_converters/cityscapes.py data/cityscapes --nproc 8

wget https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_train_anon.zip
unzip Dark_Zurich_train_anon.zip
fi

cd "$initial_dir"
### setup da former stuff
