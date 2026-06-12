from flask import Flask, render_template_string, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
model = None
label_encoders = None
scaler = None
try:
except Exception as e:
