#!/bin/bash

here=`pwd`

for aa in ala asn asp0 cys gln glu0 leu lys0 ;do
	folder="/storage/brno14-ceitec/shared/softmatter/sofbp/OK2/analogs/$aa/plot/"
	cd $folder
	prof=`ls *-moved.xvg | tail -1`
	cd $here
	cp $folder/$prof $aa-PC.xvg

	folderPE="/storage/brno14-ceitec/shared/softmatter/sofbp/OK2/analogs/PE/$aa/plot"
	cd $folderPE
	profPE=`ls *-moved.xvg | tail -1`
	cd $here
	cp $folderPE/$profPE $aa-PE.xvg
done
