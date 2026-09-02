import os

def load_env(file_path):
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split('=', 1)
            os.environ[key] = value

load_env(os.path.dirname(__file__)+"/pipeline_setup.env")
pipeline_config=os.environ

pipeline_config["pipeline_path"]=pipeline_config["grandlib_path"]+pipeline_config["pipeline_path"]
pipeline_config["python_interpreter"]=pipeline_config["conda_lib"]+pipeline_config["python_interpreter"]
pipeline_config["default_config"]=pipeline_config["pipeline_path"]+"/"+pipeline_config["default_config"]
pipeline_config["register_transfer"]=pipeline_config["pipeline_path"]+"/"+pipeline_config["register_transfer"]
pipeline_config["register_convert"]=pipeline_config["pipeline_path"]+"/"+pipeline_config["register_convert"]
pipeline_config["send_notification_email"]=pipeline_config["pipeline_path"]+"/"+pipeline_config["send_notification_email"]

pipeline_config["register_in_db"]=pipeline_config["grandlib_path"]+"/"+pipeline_config["register_in_db"]

if __name__ != "__main__":
    # This will be imported by other modules
    pass