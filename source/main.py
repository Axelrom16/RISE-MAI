from RISE import *

####
# RISE
####
cat_variables_heart = ['sex', 'chest', 'sugar', 'ecg', 'angina', 'slope', 'thal', 'disease']
cat_variables_breast = ['Class']
cat_variables_obesity = ['Gender', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE',
                             'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad']
cat_variables_toy = ['Age', 'VisualDeficiency', 'Astigmatism', 'TearProduction', 'RecommendedLense']

####
# RISE execution
####
"""
# Heart data set
model = RISE(dataset_name='heart', target_name='disease', cat_variables=cat_variables_heart,
             test_split=0.2, max_iter=10)
final_rules_heart = model.fit()
model.evaluate()

# Breast data set

model = RISE(dataset_name='breast', target_name='Class', cat_variables=cat_variables_breast,
             test_split=0.2, max_iter=5)
final_rules_breast = model.fit()
model.evaluate()
"""
# Obesity data set
"""
model = RISE(dataset_name='obesity', target_name='NObeyesdad', cat_variables=cat_variables_obesity,
             test_split=0.3, max_iter=5)
final_rules_breast = model.fit()
model.evaluate()
"""
# Toy data set

model = RISE(dataset_name='toy_dataset', target_name='RecommendedLense', cat_variables=cat_variables_toy,
             test_split=0.2, max_iter=10)
final_rules_breast = model.fit()
model.evaluate()





