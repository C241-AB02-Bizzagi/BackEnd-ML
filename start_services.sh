#!/bin/bash

# Menjalankan Celery worker di background dengan eventlet
celery -A main.celery worker -l info -P eventlet &



# Menjalankan Flask server melalui main.py
python main.py
