
pylint --output-format=colorized churn_library.py
pylint --output-format=colorized churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
black churn_library.py
black churn_script_logging_and_tests.py
pytest churn_script_logging_and_tests.py


