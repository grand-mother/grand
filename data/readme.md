# Description 

## detector directory

**Under github doesn't content data files, because they are too large**

This directory stored data defining the various models for GRANDLIB like:
 * antenna for each axis
 * electronics effect: LNA, filter, cable
 * galaxy signal 

Automatically loaded with env/setup.sh, 

# Notes about download script

## How reload a GRAND model

* remove/change name of directory grand/data/model/detector
* in directory grand/data, do 

```
python download_data_grand.py
```
