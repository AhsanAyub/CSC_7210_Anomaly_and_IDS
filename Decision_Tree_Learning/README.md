## Decision Tree for Intrusion Detection Systems (IDS)

In this project, we use the decision tree algorithm provided in Russell and Norvig’s book “Artificial Intelligence, A Modern Approach.” We build the solution using Python 3 with the help of Pandas library to access the datasets along with other libraries mentioned below, evaluate the constructed decision tree with the KDD datasets as well as a toy example dataset, and achieve satisfactory performance.

#### Execution Information
The following step needs to be taken to execute the program on Unix environment:

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

To run the algorithm with the toy example, again the following step needs to be taken by changing the arguments:

```
$ python3 decision_tree.py ids-train-example.txt ids-test-example.txt ids-attr-example.txt 
> Train, test, and attribute files are read properly
Training Set Shape: (12, 11)
Testing Set Shape: (2, 11)

============ Decision Tree ============
Patrons (Info Gain: 0.5408520829727552 and Root)
No (Info Gain: None and Condition from its parent, Patrons, is None)
Yes (Info Gain: None and Condition from its parent, Patrons, is Some)
Hungry (Info Gain: 0.2516291673878229 and Condition from its parent, Patrons, is Full)
No (Info Gain: None and Condition from its parent, Hungry, is No)
Type (Info Gain: 0.5 and Condition from its parent, Hungry, is Yes)
Yes (Info Gain: None and Condition from its parent, Type, is Burger)
No (Info Gain: None and Condition from its parent, Type, is Italian)
Fri/Sat (Info Gain: 1.0 and Condition from its parent, Type, is Thai)
No (Info Gain: None and Condition from its parent, Fri/Sat, is No)
Yes (Info Gain: None and Condition from its parent, Fri/Sat, is Yes)

============ Performance ============
Train Accuracy: 1.0000
Test Accuracy: 0.5000
```

#### Used Libraries

```
import sys
import math
import pandas as pd
import numpy as np
from typing import Dict
```