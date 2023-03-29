# RULE - SEL 

The RISE algorithm is a type of rule-based classifier first implemented by Pedro Domingos where the rules are learned by gradually generalizing instances until there is no improvement in accuracy. We implemented the algorithm in Python 3 and we evaluate it in some data sets. 

The structure of the work is 
```bash
PW1 SEL - RISE
├── data
│   ├── breast.csv
│   ├── heart.csv
│   ├── obesity.csv 
│   └── toy_dataset.csv 
├── source
│   ├── RISE.py
│   └── main.py
├── results
│   ├── heart
│   │   ├── evaluation_heart.txt
│   │   ├── rules_heart.csv
│   │   ├── rules_heart_test.csv
│   │   └── train_heart.txt
│   ├── breast 
│   │   ├── evaluation_breast.txt
│   │   ├── rules_breast.csv
│   │   ├── rules_breast_test.csv
│   │   └── train_breast.txt
│   └── toy_dataset
│       ├── evaluation_toy_dataset.txt
│       ├── rules_toy_dataset.csv
│       ├── rules_toy_dataset_test.csv
│       └── train_toy_dataset.txt
├── documentation
│   └── report.pdf
└── readme.md
```