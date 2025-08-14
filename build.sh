#!/bin/bash
set -e

# Build the package using sbuild
sbuild --chroot-mode=unshare \
	--enable-network \
	--no-clean-source \
	-d unstable --arch-all
