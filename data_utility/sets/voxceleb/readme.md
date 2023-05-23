# Voxceleb

## Setting up the environment

The following environment variables should be set before continuing:

```bash
# example values
VOXCELEB_ROOT_DIR=${PWD}/data/voxceleb/
VOXCELEB_RAW_DIR=${VOXCELEB_ROOT_DIR}/raw
VOXCELEB_EXTRACT_DIR=${VOXCELEB_ROOT_DIR}/extract
VOXCELEB_SHARD_DIR=${VOXCELEB_ROOT_DIR}/shards
VOXCELEB_META_DIR=${VOXCELEB_ROOT_DIR}/meta
```

## (Manually) collecting the data archives

I've experienced that the download links for voxceleb1/2 can be unstable.
I recommend manually downloading the dataset from the google drive link displayed 
on [https://mm.kaist.ac.kr/datasets/voxceleb/](https://mm.kaist.ac.kr/datasets/voxceleb/).

You should end up 4 zip files, which are expected to be placed in `$VOXCELEB_RAW_DIR`. 

1. `vox1_dev_wav.zip` 
2. `vox1_test_wav.zip`
3. `vox2_dev_aac.zip`
4. `vox2_test_aac.zip`


### Transforming data to wav

The voxceleb2 data needs to be converted from aac to wav format
This requires ffmpeg to be installed on the machine. Check with `ffmpeg -version`.
Assuming the voxceleb2 data archives are found at `$VOXCELEB_RAW_DIR/vox2_dev_aac.zip` 
and `$VOXCELEB_RAW_DIR/vox2_test_aac.zip`, run the following:


```bash
WORKERS=$(nproc --all) # number of CPUs available 

# extract voxceleb 2 data
mkdir -p convert_tmp/train/wav convert_tmp/test/wav

unzip $VOXCELEB_RAW_DIR/vox2_dev_aac.zip -d convert_tmp/train
unzip $VOXCELEB_RAW_DIR/vox2_test_aac.zip -d convert_tmp/test

# run the conversion script
poetry run convert_to_wav --dir convert_tmp --ext .m4a --out wav --workers $WORKERS

# move all *.wav files to wav folder instead of aac folder
# TODO

# rezip the converted data
cd convert_tmp/train
zip $VOXCELEB_RAW_DIR/vox2_dev_wav.zip wav -r

cd ../test
zip $VOXCELEB_RAW_DIR/vox2_test_wav.zip wav -r

# delete the unzipped .m4a files
cd $D
rm -r convert_tmp
```

Note that this process can take a few hours on a fast machine and day(s) on a single (slow) cpu.
Make sure to save the `vox2_dev_wav.zip` and `vox2_test_wav.zip` files somewhere secure, so you don't have redo
this process :).


## Extracting files

After the data archives are present, the next step is to extract the archive:

```bash
./voxceleb/unzip_voxceleb_archives.sh
```


## Writing shards

In sequential order:

``````
./voxceleb/default_setup_shards.sh
./voxceleb/default_write_shards.sh
./voxceleb/default_generate_meta_files.sh
```