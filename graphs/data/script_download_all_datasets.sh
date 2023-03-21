

# Command to download dataset:
#   bash script_download_all_datasets.sh



############
# ZINC
############

DIR=molecules/
cd $DIR

FILE=ZINC.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/ZINC.pkl -o ZINC.pkl -J -L -k
fi

FILE=ZINC-full.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/ZINC-full.pkl -o ZINC-full.pkl -J -L -k
fi

cd ..


############
# PATTERN and CLUSTER 
############

DIR=SBMs/
cd $DIR

FILE=SBM_CLUSTER.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/SBM_CLUSTER.pkl -o SBM_CLUSTER.pkl -J -L -k
fi

FILE=SBM_PATTERN.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/SBM_PATTERN.pkl -o SBM_PATTERN.pkl -J -L -k
fi

cd ..



############
# CSL 
############

DIR=CSL/
cd $DIR

FILE=CSL.zip
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/rnbkp5ubgk82ocu/CSL.zip?dl=1 -o CSL.zip -J -L -k
	unzip CSL.zip -d ./
	rm -r __MACOSX/
fi

cd ..



############
# CYCLES 
############

DIR=cycles/
cd $DIR

FILE=CYCLES_6_56.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/9fs9aqfp10q9wue/CYCLES_6_56.pkl?dl=1 -o CYCLES_6_56.pkl -J -L -k
fi

cd ..



############
# PLANARITY
############

DIR=planarity/
cd $DIR

FILE=Planarity.pkl
if test -f "$FILE"; then
    echo -e "$FILE already downloaded."
else
    echo -e "\ndownloading $FILE..."
    curl https://www.dropbox.com/s/l16u4xf8v58bdq1/Planarity.pkl?dl=1 -o Planarity.pkl -J -L -k
fi

cd ..


############
# CYCLES-V
############

DIR=cycles/
cd $DIR

FILE=CYCLES_-1_56.pkl
if test -f "$FILE"; then
    echo -e "$FILE already downloaded."
else
    echo -e "\ndownloading $FILE..."
    curl https://www.dropbox.com/s/e9bid1nhlgw5v32/CYCLES_-1_56.pkl?dl=1 -o CYCLES_-1_56.pkl -J -L -k
fi

cd ..

