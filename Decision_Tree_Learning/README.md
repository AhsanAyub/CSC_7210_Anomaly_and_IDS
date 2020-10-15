## Decision Tree for Intrusion Detection Systems (IDS)

In this project, we use the decision tree algorithm provided in the Russell and Norvig’s book “Artificial Intelligence, A Modern Approach.” We build the solution using Python with the help of Pandas library to access the datasets along with other libraries mentioned below, evaluate the constructed decision tree with the KDD datasets as well as a toy example dataset, and achieve the satisfactory performance.

#### Execution Information
The following steps need to be taken to execute the program:

```
$ python3 decision_tree.py ids-train.txt ids-test.txt ids-attr.txt
> Train, test, and attribute files are read properly
Training Set Shape: (800, 12)
Testing Set Shape: (200, 12)

============ Decision Tree ============
serror_rate (Info Gain: 1.0 and Root)
normal (Info Gain: None and Condition from its parent, serror_rate, is 1-24)
normal (Info Gain: None and Condition from its parent, serror_rate, is 25-49)
neptune (Info Gain: None and Condition from its parent, serror_rate, is 75-99)
neptune (Info Gain: None and Condition from its parent, serror_rate, is oneHundred)
normal (Info Gain: None and Condition from its parent, serror_rate, is zero)

============ Performance ============
Train Accuracy: 1.0000
Test Accuracy: 1.0000
```

#### Used Libraries

```
import sys
import math
import pandas as pd
import numpy as np
from typing import Dict
```