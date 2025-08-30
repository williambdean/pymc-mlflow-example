# Kick off scripts with the various examples
python 01-basic-introduction.py
python 01-basic-introduction.py
python 01-basic-introduction.py
python 01-basic-introduction.py

python 02-pymc-context.py
python 02-pymc-context.py

python 03-pymc-autologging.py nutpie normal
python 03-pymc-autologging.py nutpie normal --mock
python 03-pymc-autologging.py pymc normal
python 03-pymc-autologging.py numpyro normal
python 03-pymc-autologging.py numpyro student_t
# This will fail because of negative values in the data
python 03-pymc-autologging.py pymc gamma

export PYTHONPATH=.
python 04-pymc-marketing-mmm
