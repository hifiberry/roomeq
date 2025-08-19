#!/bin/bash
set -e

# Clean Rust build artifacts before packaging
echo "Cleaning Rust build artifacts..."
rm -rf src/rust/target/
rm -f src/rust/Cargo.lock

# Ensure we have the keyring files
if [ ! -f "hifiberry-archive-keyring.gpg" ]; then
    echo "Copying HiFiBerry keyring..."
    sudo cp /usr/share/keyrings/hifiberry-archive-keyring.gpg .
fi

if [ ! -f "hifiberry-archive-keyring.asc" ]; then
    echo "Exporting HiFiBerry key to ASCII format..."
    gpg --no-default-keyring --keyring ./hifiberry-archive-keyring.gpg --armor --export > hifiberry-archive-keyring.asc
fi

# Build the package using sbuild with HiFiBerry repository
# Use HTTP instead of HTTPS to avoid certificate issues in chroot
sbuild --chroot-mode=unshare \
	--enable-network \
	--no-clean-source \
	--extra-repository="deb http://debianrepo.hifiberry.com bookworm main" \
	--extra-repository-key=hifiberry-archive-keyring.asc \
	--chroot-setup-commands="apt-get update && apt-get install -y ca-certificates curl" \
	-d bookworm --arch-all
