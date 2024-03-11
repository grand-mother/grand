#!/bin/bash
# Script to activate/desactivate network in Grand@auger DAQ
# Need on parameter in -init and -close
# Fred & Fleg: 03/2024
# Copyright : Grand Observatory 2024

claro_hop="claro.net.ar";               # 4G operator tracepath hop
auger_hop="auger.org.ar";               # auger network tracepath hop
wwan_con="netplan-cdc-wdm0";            # NetworkManager wwan connection name
max_hop=5;                              # max hop for tracepath test
sleep_delay=2;                          # sleep delay to wait after NetworkManager (de)activation calls
verbose=false;                           # true or false

# usage: verbose_echo <string to print if verbosity is on>
verbose_echo () {
    $verbose && echo "$*";
}

wwan_activated() { 
  local wwan_state=$(nmcli c show $wwan_con | awk '/^GENERAL.STATE:/{print $2}'); 
  verbose_echo "wwan_activated - wwan_state: $wwan_state"
  test "$wwan_state" = "activated"; 
}

up_wwan() {
  nmcli c up $wwan_con
  sleep $sleep_delay
}

down_wwan() {
  nmcli c down $wwan_con
  sleep $sleep_delay
}

in2p3_route_claro() {
  local tracepath_output;
  tracepath_output=$(tracepath -m $max_hop $in2p3_machine)
  verbose_echo "in2p3_route_claro - tracepath_output: $tracepath_output" 
  echo $tracepath_output | grep -q $claro_hop;
}

in2p3_route_auger() {
  local tracepath_output;
  tracepath_output=$(tracepath -m $max_hop $in2p3_machine)
  verbose_echo "in2p3_route_auger - tracepath_output: $tracepath_output" 
  echo $tracepath_output | grep -q $auger_hop;
}

switch_on() {
	# switch on interface if not activated
	if ! wwan_activated ; then
	  up_wwan
	elif in2p3_route_auger ; then  # interface already activated but still auger route
	  # up/down cycle
	  down_wwan
	  up_wwan
	fi

	# exit if wwan still not activated
	if ! wwan_activated ; then
	  echo "cannot activate $wwan_con";
	  down_wwan
	  exit 1
	fi

	# wwan supposed to be activated, test if route to in2p3 is still through auger network
	if in2p3_route_auger ; then
	  echo "wrong route to $in2p3_machine, still going through auger network"
	  down_wwan
	  exit 1
	fi
}

# Main
if [ "$#" -ne 1 ]
then
        echo "Incorrect number of arguments : 1 needed, ${#} given"
        exit 2
else
        if [ "$1" = "-init" ]; then
                switch_on
        elif [ "$1" = "-close" ]; then
                down_wwan
        else
                echo "Bad option"
		exit 2
        fi
fi



