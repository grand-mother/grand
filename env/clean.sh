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

    local prefix="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

    local tag="$(basename ${prefix})"
    logmsg () {
        [[ -z "${quiet}" ]] && echo "[${tag}] $@"
    }

    logmsg "--Cleaning the environment"

    remove_path "bin"
    remove_path "user/grand/.local/bin"
    remove_pythonpath "lib/python"

    logmsg "--Environment cleaned"
}


remove_path () {
    local path="${prefix}/${1}"
    if [[ "$PATH" =~ "${path}" ]]; then
        logmsg "  Removing \$PREFIX/${1} from PATH"
        local work=":${PATH}:"
        local remove="${path}"
        work="${work/:$remove:/:}"
        work="${work%:}"
        work="${work#:}"
        export PATH="${work}"
    fi
}


remove_pythonpath () {
    local path="${prefix}/${1}"
    if [[ "$PYTHONPATH" =~ "${path}" ]]; then
        logmsg "  Removing \$PREFIX/${1} from PYTHONPATH"
        local work=":${PYTHONPATH}:"
        local remove="${path}"
        work="${work/:$remove:/:}"
        work="${work%:}"
        work="${work#:}"
        export PYTHONPATH="${work}"
    fi
}


main "$@"
unset -f logmsg main remove_path remove_pythonpath
