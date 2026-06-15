"""Wrapper to run the Week 2 pipeline on the full dataset.

This sets the MAX_ROWS env var to disable subsampling and calls the run_week2 main().
"""
import os
os.environ['MAX_ROWS'] = 'None'

from Week_2_Fraud_Detection.run_week2 import main

if __name__ == '__main__':
    main()
