#!/usr/bin/env bash
set -euo pipefail

PX4_DIR="${1:?usage: install_px4_airframe.sh /path/to/PX4-Autopilot}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$HERE/../px4/4002_gz_x500_swarm"

if [[ ! -f "$SOURCE" ]]; then
  echo "airframe source not found: $SOURCE" >&2
  exit 1
fi
if [[ ! -d "$PX4_DIR" ]]; then
  echo "PX4 directory not found: $PX4_DIR" >&2
  exit 1
fi

install -m 644 "$SOURCE" "$PX4_DIR/ROMFS/px4fmu_common/init.d-posix/airframes/4002_gz_x500_swarm"
install -m 644 "$SOURCE" "$PX4_DIR/build/px4_sitl_default/etc/init.d-posix/airframes/4002_gz_x500_swarm"
echo "installed 4002_gz_x500_swarm into $PX4_DIR"
