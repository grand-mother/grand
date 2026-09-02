.. This page is generated from resources/GRANDlib_Handbook.zip
   by docs/dev/build_handbook.py.  Do not edit it by hand.

Grandlib Classes
================


To be able to understand better the Analysis and Data Oriented Interface (AOI,DOI) you can explore a jupyter notebook that has been created.

You can start by downloading a `docker image <https://drive.google.com/file/d/1OYf_LEt1H0qKIjqZ6bGcW6P8gUW9JBtB/view?usp=drive_link>`__

The image is very big (almost 5 GB), so please download before session.

Load the docker tar into the local repository:

.. code:: bash

   docker load -i grand_docker_handson_2025.tar.gz 

Then run the docker image with:

.. code:: bash

   docker run -it -v "$HOME/GRAND:/opt/GRAND:Z" -p 8888:8888 grandlib-dev 

You can remove -v "$HOME/GRAND:/opt/GRAND:Z" completely, or you need to replace it with a local directory that you want to access in docker.

Inside the docker, execute:

.. code:: bash

   ./run_me

to run the hands-on jupyter notebook. Then access it with your local web browser by accessing https://localhost:8888

1) If you have a working GRANDlib

#. Be sure it is updated from the dev branch

#. clone `https://github.com/grand-mother/hands-on_software_warsaw_2025 <https://github.com/grand-mother/hands-on_software_warsaw_2025/blob/main/hands-on_2025.ipynb>`__

#. Download the data directories and unpack in the hands-on directory
   `DATA1 <https://drive.google.com/file/d/1IpzxE7s0Uff2q9uQDmPOOseYPVmbtjs_/view?usp=drive_link>`__
   `DATA2 <https://drive.google.com/file/d/1VqUi4NO1vBzXrFKgcv67gNlWbAmEorTj/view?usp=drive_link>`__

2) If you don’t have linux

I’m afraid that then you need to access linux through a virtual machine, such as VirtualBox. You can download whatever linux you want to use, for example `Fedora 41 <https://sourceforge.net/projects/osboxes/files/v/vb/18-F-d/41/Workstation/64bit.7z/download>`__

Then install docker inside and follow the initial instructions.

For additional examples on viewing and analyzing ROOT files, please refer to the Examples directory.

