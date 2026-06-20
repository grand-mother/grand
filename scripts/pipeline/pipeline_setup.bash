#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
source ${SCRIPT_DIR}/pipeline_setup.env
pipeline_path="${grandlib_path}${pipeline_path}"
python_interpreter="${conda_lib}/${python_interpreter}"
default_config="${pipeline_path}/${default_config}"
register_transfer="${pipeline_path}/${register_transfer}"
register_convert="${pipeline_path}/${register_convert}"
register_in_db="${grandlib_path}/${register_in_db}"
send_notification_email="${pipeline_path}/${send_notification_email}"