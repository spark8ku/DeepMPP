import os
import importlib

# Allowing to access all modules in the directory

current_dir = os.path.dirname(os.path.abspath(__file__))

module_files = [f for f in os.listdir(current_dir) if f.endswith('.py') and f != '__init__.py']

for module_file in module_files:
    module_name = module_file[:-3]  
    module = importlib.import_module(f'.{module_name}', package='src.Analyzer')  
    
    for attr in dir(module):
        if not attr.startswith('_'):  
            globals()[attr] = getattr(module, attr)
