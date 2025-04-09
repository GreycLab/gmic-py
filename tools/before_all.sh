#!/usr/bin/env bash
# CI build script, designed to be run on a RHEL-based or Alpine-based image

set -e

md5sum "$0" > /tmp/before_all_hash.txt

if ! diff -Nq /tmp/before_all_hash.txt /var/lib/before_all_hash.txt &>/dev/null; then
	if which dnf &>/dev/null; then
		dnf install -y libX11-devel libXext-devel libtiff-devel libpng-devel libjpeg-devel fftw-devel \
		  OpenEXR-devel ilmbase-devel zlib-devel opencv-devel GraphicsMagick-c++-devel libcurl-devel
	elif which apk &>/dev/null; then
		apk add libpng-dev fftw-dev zlib-dev opencv-dev graphicsmagick-dev jpeg-dev tiff-dev openexr-dev curl-dev
	else
		echo "Unrecognized platform - couldn't install dependencies" >&2
		exit 1
	fi
	mv -v /tmp/before_all_hash.txt /var/lib/
else
	echo "Skipping before_all: up-to-date"
fi
