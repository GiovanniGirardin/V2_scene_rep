from __future__ import annotations

import os

print("SUMO_HOME:", os.environ.get("SUMO_HOME"))

import smarts
import sumolib

print("SMARTS:", smarts)
print("SUMO:", sumolib)

scenario = "/home/giovanni/SMARTS/scenarios/sumo/loop"
print("Scenario:", scenario)
print("SMARTS import OK")
