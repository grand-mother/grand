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

    local prefix="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
    # add to env variable define root pf package
    export GRAND_ROOT=$prefix
    local tag="$(basename ${prefix})"
    logmsg () {
        [[ -z "${quiet}" ]] && echo "[${tag}] $@"
    }


    logmsg "--Setting the environment"

    # Expand the PATH
    expand_path () {
        local path="${prefix}/${1}"
        if [[ ! "$PATH" =~ "${path}" ]]; then
            logmsg "  Adding \$PREFIX/${1} to PATH"
            export PATH="${path}:${PATH}"
        fi
    }

    # add in PATH quality for script
    expand_path "quality"

    expand_pythonpath () {
        local path="${prefix}/${1}"
        if [[ ! "$PYTHONPATH" =~ "${path}" ]]; then
            logmsg "  Adding \$PREFIX/${1} to PYTHONPATH"
            export PYTHONPATH="${path}:${PYTHONPATH}"
        fi
    }

    expand_pythonpath ""

    # Build the C extensions
	cd $GRAND_ROOT/src
	install_ext_lib.bash
	cd -
	
    logmsg "--Environment set"
}


main "$@"
unset -f expand_path logmsg main sanitize_shebangs
