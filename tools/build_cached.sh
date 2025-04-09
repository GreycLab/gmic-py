#!/bin/bash

DIR=$(dirname "$0")

MANYLINUX_IMAGE="gmic-py-manylinux"
MUSLLINUX_IMAGE="gmic-py-musllinux"
export CIBW_MUSLLINUX_X86_64_IMAGE="$MUSLLINUX_IMAGE"
export CIBW_MUSLLINUX_AARCH64_IMAGE="$MUSLLINUX_IMAGE"
(
	if [ "$1" = "build" ] || ! docker inspect "$MANYLINUX_IMAGE" &>/dev/null; then
		if read -r MANYLINUX_BASE < <(
			sed -rn 's/^manylinux-x86_64-image *= *"(.*)".*$/\1/p' "$DIR/../pyproject.toml" ); then
			echo "Building $MANYLINUX_IMAGE from $MANYLINUX_BASE"
			docker build "$DIR" --build-arg IMGFROM="$MANYLINUX_BASE" -t "$MANYLINUX_IMAGE"
		else
			echo "Couldn't read manylinux base image from pyproject.toml" >&2
			exit 1
		fi
	fi
) && \
	export CIBW_MANYLINUX_X86_64_IMAGE="$MANYLINUX_IMAGE" && \
	export CIBW_MANYLINUX_AARCH64_IMAGE="$MANYLINUX_IMAGE"

(
	if [ "$1" = "build" ] || ! docker inspect "$MUSLLINUX_IMAGE" &>/dev/null;then
		if read -r MUSLLINUX_BASE < <(
			sed -rn 's/^musllinux-x86_64-image *= *"(.*)".*$/\1/p' "$DIR/../pyproject.toml" ); then
			echo "Building $MUSLLINUX_IMAGE from $MUSLLINUX_BASE"
			docker build "$DIR" --build-arg IMGFROM="$MUSLLINUX_BASE" -t "$MUSLLINUX_IMAGE"
		else
			echo "Couldn't read musllinux base image from pyproject.toml" >&2
			exit 1
		fi
	fi
) && \
	export CIBW_MUSLLINUX_X86_64_IMAGE="$MUSLLINUX_IMAGE" && \
	export CIBW_MUSLLINUX_AARCH64_IMAGE="$MUSLLINUX_IMAGE"

env | grep '^CIBW_'
