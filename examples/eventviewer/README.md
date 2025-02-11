This event-viewer package is written to view various properties of cosmic ray extensive air showers detected by the GRAND.
"holoviews" python package is used to create this event-viewer. It uses 'bokeh' and 'matplotlib' as a base for visualization and creating widget.

Needs python 3.6 or higher.

Install holoviews using pip3. All required dependencies will automatically be installed, including 'bokeh'.
For more info: http://holoviews.org/

  	$ pip3 install "holoviews[all]" 
  
  
After installing holoviews, add $HOME/.local/bin in the PATH so that packages installed there can be accessed.

  	$ cd ~
  
  	$ emacs -nw .bashrc      (or .bash_profile)
  
Inside .bashrc (or .bash_profile), add the following line

    export PATH=$HOME/.local/bin:$PATH


If above mentioned process fails to install 'bokeh', then install it by using pip3. If not, skip this part.
For more info: https://docs.bokeh.org/en/latest/

  	$ pip3 install bokeh 


After successfully installing "holoviews", install event-viewer in a directory you like as follows.
  
  	$ git clone https://github.com/rameshkoirala/EventViewer.git
  
After successfully installing EventViewer, you can run the package by

  	$ python3 event_viewer.py --datadir <path>
	
  	$ python3 event_viewer.py --datadir <path> --hf <filename> --gf <geometry filename>

A window of your default web-browser will pop-up showing features about the event (default or from --hf <file>). 
"Play" widget is to play the selected event. "Browse" widget is to select a new event file. 
Use features and widgets of the event-viewer to visualize cosmic ray air shower events. Enjoy !!!


----------- MIT License -------------
	
Copyright (c) 2021 GRAND Collaboration
	
contact: rkoirala@nju.edu.cn (Ramesh Koirala)

	
Permission is hereby granted, free of charge, to any person obtaining a copy of this 
software and associated documentation files (the "Software"), to deal in the Software 
without restriction, including without limitation the rights to use, copy, modify, merge,	
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons	
to whom the Software is furnished to do so, subject to the following conditions:


The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. 
	

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
