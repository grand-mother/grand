#! /bin/bash

main () {
    local OPTIND=1 option quiet
    while getopts "q" option
    do
    case "${option}"
    in
        q) quiet=true;;
    esac
    done

    local prefix="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

    local tag="$(basename ${prefix})"
    logmsg () {
        [[ -z "${quiet}" ]] && echo "[${tag}] $@"
    }

    if [ ! -e "${prefix}/bin/python3-$(arch).AppImage" ]; then
        logmsg "--Fetching AppImage"
        local url="https://github.com/grand-mother/python/releases/download/continuous/python3-$(arch).AppImage"
        (cd "${prefix}/bin" && wget -cq "${url}")
        if [ "$?" != "0" ]; then
            logmsg "  could not fetch ${url}"
            exit 1
        fi
        logmsg "  AppImage retrieved"
    fi
    chmod u+x "${prefix}/bin/python3-$(arch).AppImage" 

    if [ ! -d "${prefix}/user/grand" ]; then
        mkdir -p "${prefix}/user/grand"
    fi

    logmsg "--Setting the environment"

    # Expand the PATH
    expand_path () {
        local path="${prefix}/${1}"
        if [[ ! "$PATH" =~ "${path}" ]]; then
            logmsg "  Adding ${1} to PATH"
            export PATH="${path}:${PATH}"
        fi
    }

    expand_path "bin"
    expand_path "user/grand/.local/bin"

    # Rebase the shebangs of Python scripts
    sanitize_shebangs () {
        local exe
        for exe in $(ls -d "${prefix}/${2}/"* 2>/dev/null)
        do
            [ -d "${exe}" ] && continue
            if [ "$(sed -n '/^#!.*\/python.*/p;q' ${exe})" ]; then
                local name=$(basename "${exe}")
                logmsg "  Sanitizing ${1}::$name"
                sed -i '1s|^#!.*\/python.*|#!/usr/bin/env python|' "$exe"
            fi
        done
    }

    sanitize_shebangs "system::" "bin"
    sanitize_shebangs "user::" "user/grand/.local/bin"

    logmsg "--Environment set"
}


main "$@"
unset -f expand_path logmsg main sanitize_shebangs
