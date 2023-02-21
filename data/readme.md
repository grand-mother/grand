# Description 

## model directory

This directory stored data defining the various models for GRANDLIB like:
 * antenna for each axis
 * electronics effect: LNA, filter, cable
 * galaxy signal 

Automatically loaded with env/setup.sh, **so under github doesn't content data files**

## tests directory

Directory to automatic test

# Notes about download script

## How reload a GRAND model

* remove/change name of directory grand/data/model/detector
* in directory grand/data, do 

```
python download_data_grand.py
```
