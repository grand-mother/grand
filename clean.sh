#!/bin/bash

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

    logmsg "--Cleaning the environment"

    remove "bin"
    remove "user/grand/.local/bin"

    logmsg "--Environment cleaned"
}


remove () {
    local path="${prefix}/${1}"
    if [[ "$PATH" =~ "${path}" ]]; then
        logmsg "  Removing ${1} from PATH"
        local work=":${PATH}:"
        local remove="${path}"
        work="${work/:$remove:/:}"
        work="${work%:}"
        work="${work#:}"
        export PATH="${work}"
    fi
}


main "$@"
unset -f logmsg main remove
