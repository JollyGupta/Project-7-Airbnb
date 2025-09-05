# system_report.py

def system_and_library_report():
    import sys
    import os
    import platform
    import pandas as pd
    import numpy as np
    import sklearn
    import matplotlib
    import seaborn as sns
    import joblib

    print("Library Versions:")
    print(f"  pandas:        {pd.__version__}")
    print(f"  numpy:         {np.__version__}")
    print(f"  scikit-learn:  {sklearn.__version__}")
    print(f"  matplotlib:    {matplotlib.__version__}")
    print(f"  seaborn:       {sns.__version__}")
    print(f"  joblib:        {joblib.__version__}")
    
    print("\n Python Version:")
    print(f"  {sys.version}")
    
    print("\n System Info:")
    print(f"  OS:            {platform.system()}")
    print(f"  OS Version:    {platform.version()}")
    print(f"  Machine:       {platform.machine()}")
    print(f"  Processor:     {platform.processor()}")
    print(f"  Architecture:  {platform.architecture()[0]}")
    print(f"  CPU Count:     {os.cpu_count()}")
