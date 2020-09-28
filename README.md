## Toolkit for MorphTrack

### requirements
The MATLAB code depends on [BioFormat](http://www.openmicroscopy.org/bio-formats/downloads/). Folder *MATLAB_Lib* should be added in MATLAB path.
### label-tool
A user-friendly-interface to label the [raw data](https://figshare.com/articles/dataset/raw-image-data/12792554). Typical usage:
```MATLAB
[fn,fp] = uigetfile('*.dv'); %GET DV FILE PATH
dv = DVImageReader(strcat(fp,fn)); %GET DV FILE INFO
dv.parse %LOAD DATA INTO RAM
makeMask(dv) %OPEN THE GUI
```
<div style="align: center">
<img src="https://s1.ax1x.com/2020/09/28/0AWorj.png" width = "500" height = "450"/>
</div>

Hit **initial estimation** button and select **AutoContour** and hit **yes** will give a initial estimation of the mask.

<div style="align: center">
<img src="https://s1.ax1x.com/2020/09/28/0AWTqs.png" width = "500" height = "450"/>
</div>

Then you can manually adjust the mask by bush mode. Note that the **Mix Ratio** need to be adjusted to show the mask. The edge can be estimated by hit **NFE(new from edge)** button.
## Mask-Tracker
MaskTracker track cell mask frame-by-frame.
```MATLAB
[fn,fp] = uigetfile('*.dv'); %GET DV FILE PATH
dv = DVImageReader(strcat(fp,fn)); %GET DV FILE INFO
dv.parse %LOAD DATA INTO RAM
load('HeLa_tag.mat') %LOAD TAG FILE
tags = permute(tags+1,[1,3,2]); %FORMET AS NHW, MASK ID SHOULD EQUAL TO 2
maskTracker(dv,tags); %OPEN GUI
```

<div style="align: center">
<img src="https://s1.ax1x.com/2020/09/28/0AI641.png" width = "600" height = "300"/>
</div>

## Triplet Autoencoder
Please run to **triplet_ae.py**.
## About
Please email zhaohs16@mails.tsinghua.edu.cn for any question about the code and paper
