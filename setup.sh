#!/usr/bin/env bash
# Nocturna setup: checks this machine, installs what is missing and sets Nocturna up.
# docs/HOME_SERVER.md explains every step. Safe to run again at any time.
#
#   sudo ./setup.sh            check, show what will change, ask, then set it up
#   sudo ./setup.sh --check    report only; changes nothing
#
# It needs root to install packages and to use Docker, which this setup runs
# with sudo instead of adding anyone to the docker group.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ${EUID} -ne 0 ]]; then
    echo "The setup needs root (for apt, Docker and Tailscale). Asking sudo:"
    exec sudo bash "${here}/setup.sh" "$@"
fi

if ! command -v python3 >/dev/null 2>&1; then
    case " $* " in
        *" --check "*)
            echo "python3 is not installed; the setup installs it when run without --check."
            exit 1 ;;
    esac
    echo "Installing python3, which the setup is written in..."
    apt-get -o DPkg::Lock::Timeout=300 update
    DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 install -y python3
fi

# -I: ignore the environment's Python settings and the user's site packages
exec python3 -I "${here}/setup/installer.py" "$@"
