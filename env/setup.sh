#! /bin/bash
#
# Sets up a GRANDlib working environment: search paths, the compiled C
# extensions, and the data model.
#
#     source env/setup.sh
#
# Source it, do not execute it: it exports variables the calling shell needs.
#
# It deliberately does NOT use `set -e`.  Being sourced, that would leak into
# the caller's interactive shell and abort it on the next failing command --
# closing a terminal because a `grep` found nothing.  Each step's status is
# checked explicitly instead, and the whole script returns non-zero if any
# failed.
#
# That matters more than it sounds.  This script used to end on a `cd`, so its
# status was the `cd`'s: a failed C-extension build reported success, and the
# failure surfaced later and somewhere unrelated, as
# "ModuleNotFoundError: No module named 'grand._core'".

call_path=$PWD
script_full_path=$(dirname "${BASH_SOURCE[0]}")

# `return` is only valid in a sourced script; `exit` would close the caller's
# shell.  Detect which we are so the status can be reported either way.
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    _grand_sourced=1
else
    _grand_sourced=0
fi

_grand_status=0
_grand_failed=""

# Records a failure without aborting, so that later steps still run and the
# user sees everything that is wrong in one pass rather than one per attempt.
_grand_check() {
    local rc=$1
    local label=$2
    if [ "$rc" -ne 0 ]; then
        _grand_status=$rc
        _grand_failed="${_grand_failed}
  - ${label} (exit ${rc})"
    fi
}

cd "$script_full_path" || return 1 2>/dev/null || exit 1

. ./_setup_env.sh
_grand_check $? "environment variables (env/_setup_env.sh)"

. ./_setup_lib.sh
_grand_check $? "compiling TURTLE and GULL (src/install_ext_lib.bash)"

_setup_vs_code.py
_grand_check $? "writing the VS Code environment file"

# download data model for GRAND
data/download_data_grand.py
_grand_check $? "downloading the data model (data/download_data_grand.py)"
#data/download_new_RFchain.py

cd "$call_path"

if [ "$_grand_status" -ne 0 ]; then
    echo ""
    echo "=============================================="
    echo "env/setup.sh FAILED. These steps did not succeed:${_grand_failed}"
    echo ""
    echo "The environment is incomplete; later imports will fail in ways that"
    echo "do not name this as the cause. Fix the above before continuing."
    echo "=============================================="
    echo ""
fi

unset _grand_check _grand_failed
if [ "$_grand_sourced" -eq 1 ]; then
    unset _grand_sourced
    return $_grand_status
else
    unset _grand_sourced
    exit $_grand_status
fi
