import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import timedelta
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Main class
class RISE:

    def __init__(self, dataset_name, target_name, cat_variables, test_split, max_iter):
        self.class_names = None
        self.num_classes = None
        self.attr_class = None
        self.attr_names = None
        self.name = dataset_name
        self.X = None
        self.target_name = target_name
        self.cat_variables = cat_variables
        self.n = None
        self.p = None
        self.X_test = None
        self.test_split = test_split
        self.set_rules = None
        self.out_file = None
        self.svdm_probs = None
        self.max_iter = max_iter
        self.num_max = None
        self.num_min = None

    def import_data(self):
        """
        Import the data set using the dataset_name parameter, process the data and perform the train/test split
        :return:
        """
        data = pd.read_csv('./data/' + self.name + '.csv', na_values=['?'])

        # Missing data
        if data.isnull().values.any():
            col_missing = data.columns[np.where(data.isnull().sum() > 0)]
            for col in col_missing:
                if col not in self.cat_variables:
                    data[col].fillna((data[col].mean()), inplace=True)
                else:
                    data = data.fillna(data.mode().iloc[0])
        # Encode categorical variables
        le = LabelEncoder()
        for cat in self.cat_variables:
            data[cat] = le.fit_transform(data[cat])
            print(dict(zip(le.classes_, range(len(le.classes_)))))

        # Values type
        for cat in self.cat_variables:
            data[cat] = data[cat].astype(object)

        # Train/Test split
        y = data[self.target_name]
        X = data.drop(self.target_name, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_split, random_state=42)
        X_train[self.target_name] = y_train
        X_train = X_train.reset_index(drop=True)
        X_test[self.target_name] = y_test
        X_test = X_test.reset_index(drop=True)

        self.X = X_train
        self.X = self.X[[col for col in self.X.columns if col != self.target_name] + [self.target_name]]
        self.attr_names = self.X.columns
        self.attr_class = list(self.X.dtypes)
        self.num_classes = len(np.unique(self.X[self.target_name]))
        self.class_names = np.unique(self.X[self.target_name])
        self.X_test = X_test
        self.X_test = self.X_test[[col for col in self.X.columns if col != self.target_name] + [self.target_name]]
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]

        # Write info data
        self.out_file.write("#-------\n")
        self.out_file.write("Data\n")
        self.out_file.write("#-------\n")
        self.out_file.write("Train data shape: " + str(self.X.shape) + "\n")
        self.out_file.write("Test data shape: " + str(self.X_test.shape) + "\n")

    def create_rule(self, x):
        """
        Create a vector that represents a rule given an instance
        :param x: instance
        :return: vector that represents a rule
        """
        rule = []
        for i, val in enumerate(x):
            if self.attr_class[i] != 'O':
                rule.append(f"{val} - {val}")
            else:
                rule.append(str(val))
        return rule

    def initialize_rules(self):
        """
        Initialize the set of rules, where each instance is a rule
        :return: list of rules
        """
        self.set_rules = np.array([self.create_rule(row.values) for _, row in self.X.iterrows()])
        return self.set_rules

    def estimate_SVDM_probs(self):
        """
        Estimate the probabilities used in SVDM
        """
        categorical_class_probability_dict = dict()
        indx_cat = np.arange(self.p)[np.array([self.attr_class[i] == 'O' for i in range(self.p)])]
        y = self.X[self.target_name]

        # top level dict is indexed by categorical columns
        for column in indx_cat:

            column_dict = dict()

            # second level is for the column's categories
            for category in self.X.iloc[:, column].unique():

                category_dict = dict()

                category_series = y[self.X.iloc[:, column] == category]

                total_count = category_series.count()

                # last level associates the category with a probability for each class of y
                for y_class in y.T.squeeze().unique():
                    category_dict[y_class] = category_series[category_series == y_class].count() / total_count

                column_dict[category] = category_dict

            categorical_class_probability_dict[column] = column_dict

        self.svdm_probs = categorical_class_probability_dict

    def delta_num_constants(self):
        """
        Initialize the max and min constants for the numerical distance
        """
        self.num_max = self.X.max(axis=0).tolist()
        self.num_min = self.X.min(axis=0).tolist()

    def SVDM(self, r, e, i):
        """
        Calculate the simplified value difference metric for categorical attributes
        :param r: value of the attribute in the rule
        :param e: value of the attribute in the instance
        :param i: index of the attribute
        :return: simplified value difference metric
        """
        r = float(r)
        attr_dict = self.svdm_probs[i]

        attr_dict_r = np.fromiter(attr_dict[r].values(), dtype=float)
        try:
            attr_dict_e = np.fromiter(attr_dict[e].values(), dtype=float)
        except Exception as e:
            attr_dict_e = np.zeros(len(attr_dict_r))

        value = np.sum(np.abs(attr_dict_r - attr_dict_e))

        return value

    def delta_num(self, r, e, i):
        """
        Calculate the normalized numeric range distance for numerical attributes
        :param r: value of the attribute in the rule
        :param e: value of the attribute in the instance
        :param i: index of the attribute i
        :return: normalized numeric range distance
        """
        lower, upper = map(float, r.split('-'))

        #e_max = self.X[self.attr_names[i]].max()
        e_max = self.num_max[i]
        #e_min = self.X[self.attr_names[i]].min()
        e_min = self.num_min[i]
        value = 0
        if lower <= e <= upper:
            value += 0
        elif e > upper:
            value += (e - upper) / (e_max - e_min)
        else:
            value += (lower - e) / (e_max - e_min)

        return value

    def distance(self, r, e):
        """
        Calculate the distance between a rule and an instance
        :param r: rule
        :param e: instance
        :return: distance value
        """
        value = 0
        for k in range(self.p - 1):
            if r[k] == 'None':
                return 999
            elif r[k] == 'True' or r[k] == 'T':
                value += 0
            elif self.attr_class[k] == 'O' and r[k] != 'True':
                value += self.SVDM(r[k], e[k], k)
            else:
                value += self.delta_num(r[k], e[k], k)

        return value

    def distance_2(self, e, r):
        """
        Calculate the distance between a rule and an instance
        :param r: rule
        :param e: instance
        :return: distance value
        """
        value = 0
        for k in range(self.p - 1):
            if r[k] == 'None':
                return 999
            elif r[k] == 'True' or r[k] == 'T':
                value += 0
            elif self.attr_class[k] == 'O' and r[k] != 'True':
                value += self.SVDM(r[k], e[k], k)
            else:
                value += self.delta_num(r[k], e[k], k)

        return value

    def classify(self, rs, e):
        """
        Classify an instance with a specific set of rules
        :param rs: set of rules
        :param e: instance
        :return: predicted label for the instance
        """
        #rs = [x for x in rs if 'None' not in x]
        distances = np.apply_along_axis(self.distance, axis=1, arr=rs, e=e)
        nearest_rule_index = np.argmin(distances)
        nearest_rule = rs[nearest_rule_index]
        predicted_class = nearest_rule[-1]

        return float(predicted_class)

    def estimate_accuracy(self, init, rules=None):
        """
        Estimate the accuracy of the set of rules
        :param init: if initial accuracy estimation (leave-one-out)
        :param rules: set of rules (if None, use actual set of rules)
        :return: accuracy estimation
        """
        if rules is None:
            set_rules = self.set_rules
        else:
            set_rules = rules
        if init:
            subset_rules_arr = np.array([np.delete(set_rules, i, axis=0) for i in np.arange(len(set_rules))])
            predicted = [self.classify(subset_rules_arr[i, ...], row) for i, row in self.X.iterrows()]
            c = np.count_nonzero(predicted == self.X[self.target_name])
        else:
            predicted = [self.classify(set_rules, row) for i, row in self.X.iterrows()]
            c = np.count_nonzero(predicted == self.X[self.target_name])

        acc = c / self.n
        return acc

    def rule_cover_instance(self, e, r):
        """
        Check if rule cover the instance
        :param e: instance
        :param r: rule
        :return: boolean
        """
        for i, cond in enumerate(r):
            if r[i] == True or r[i] == 'True' or r[i] == 'T':
                continue
            if self.attr_class[i] == 'O':
                if float(r[i]) != e[i]:
                    return False
            else:
                lower, upper = map(float, r[i].split('-'))
                if e[i] < lower or e[i] > upper:
                    return False
        return True

    def nearest_instance_to_rule(self, r):
        """
        Find the nearest instance to the rule r not covered by the rule and being of the same class
        :param r: rule
        :return: nearest instance
        """
        if 'None' in r:
            return None

        X = self.X.values
        target_class = float(r[-1])
        mask = np.logical_and(~np.apply_along_axis(self.rule_cover_instance, axis=1, arr=X, r=r), X[:, -1] == target_class)
        if np.sum(mask) == 0:
            return None
        distances = np.apply_along_axis(self.distance_2, axis=1, arr=X[mask, :], r=r)
        nearest_index = np.argmin(distances)
        return X[mask][nearest_index]

    def rule_generalization(self, r, e):
        """
        Create a generalized rule
        :param r: rule to be generalized
        :param e: nearest instance to the rule
        :return: new rule
        """
        if e is None:
            return None

        new_rule = []
        for i, cond in enumerate(r):
            if r[i] == True or r[i] == 'True' or r[i] == 'T':
                new_rule.append(True)
            elif self.attr_class[i] == 'O' and float(r[i]) == e[i]:
                new_rule.append(cond)
            elif self.attr_class[i] != 'O':
                lower, upper = map(float, r[i].split('-'))
                if e[i] < lower:
                    lower = e[i]
                elif e[i] > upper:
                    upper = e[i]
                new_rule.append(f"{lower} - {upper}")
            else:
                new_rule.append(True)

        return new_rule

    def fit(self):
        """
        Fit function to train the RISE rule-based classifier.
        :return: final set of rules
        """
        # Open fit log
        results_file = open("results/train_" + self.name + ".txt", "w")
        self.out_file = results_file

        # Import data
        self.import_data()

        # Initialize SVDM probs
        self.estimate_SVDM_probs()
        self.delta_num_constants()

        # Estimate initial rules (ES = RS)
        self.initialize_rules()

        self.out_file.write("\n#-------\n")
        self.out_file.write("Training\n")
        self.out_file.write("#-------\n")
        # Initial accuracy
        print("\nEstimating initial accuracy...")
        initial_accuracy = self.estimate_accuracy(init=True)
        print("Accuracy: ", initial_accuracy)
        self.out_file.write("Initial accuracy: " + str(initial_accuracy) + "\n")
        acc = initial_accuracy
        acc_increased = True
        c = 0
        start_time = time.monotonic()
        while acc_increased and c < self.max_iter:
            print("Epoch: ", str(c))
            self.out_file.write("Epoch: " + str(c) + "\n")
            for i, r in tqdm(enumerate(self.set_rules), total=len(self.set_rules)):
                new_set_rules = np.copy(self.set_rules)
                repeated_rule = any((r == x).all() for x in self.set_rules[:i])
                if repeated_rule:
                    new_set_rules[i] = None
                else:
                    # Nearest instance that does not respect all the conditions of the rule
                    instance = self.nearest_instance_to_rule(r)
                    # Generalize the rule to cover the instance
                    new_rule = self.rule_generalization(r, instance)
                    new_set_rules[i] = new_rule
                # New accuracy
                new_acc = self.estimate_accuracy(init=True, rules=new_set_rules)
                if new_acc >= acc:
                    acc = new_acc
                    # Check if the new rule already exists
                    if not repeated_rule:
                        repeated_new = any((new_rule == x).all() for x in self.set_rules)
                        if repeated_new:
                            new_set_rules[i] = None
                    # Add the generalized rule and remove the old one
                    self.set_rules = new_set_rules
                    self.out_file.write("Improved accuracy: " + str(new_acc) + "\n")
            acc_increased = acc > initial_accuracy
            initial_accuracy = acc
            c += 1

        end_time = time.monotonic()
        self.out_file.write("\nTraining time: " + str(timedelta(seconds=end_time - start_time)))
        self.out_file.close()
        return self.set_rules

    def print_rules(self):
        """
        Print actual set of rules in order to make them interpretable
        :return: list with the rules
        """
        rs = [x for x in self.set_rules if 'None' not in x]
        rule_text_vec = []
        for i, rule in enumerate(rs):
            rule_text = ''
            for j in range(self.p):
                if j == self.p - 1:
                    new = self.attr_names[j] + ' = ' + str(rule[j])
                    rule_text = rule_text + ' THEN (' + new + ')'
                    continue
                else:
                    if rule[j] == True or rule[j] == 'True' or rule[j] == 'T':
                        continue
                    elif self.attr_class[j] != 'O':
                        lower, upper = map(float, rule[j].split('-'))
                        new = str(lower) + ' <= ' + self.attr_names[j] + ' <= ' + str(upper)
                        if len(rule_text) <= 1:
                            rule_text = rule_text + '(' + new + ')'
                        else:
                            rule_text = rule_text + ' AND (' + new + ')'
                    else:
                        new = self.attr_names[j] + ' = ' + str(rule[j])
                        if len(rule_text) <= 1:
                            rule_text = rule_text + '(' + new + ')'
                        else:
                            rule_text = rule_text + ' AND (' + new + ')'
                rule_text += ' //'
            rule_text_vec.append(rule_text)
        return rule_text_vec

    def predict(self, X):
        """
        Predict class label for a given data set using the actual set of rules
        :param X: data set
        :return: predicted class
        """
        return [self.classify(self.set_rules, row) for _, row in X.iterrows()]

    def rule_coverage(self, r):
        """
        Calculate the coverage of a rule
        :param r: rule
        :return: coverage of the rule r
        """
        cover_vec = [self.rule_cover_instance(row, r) for _, row in self.X.iterrows()]
        cover_vec_test = [self.rule_cover_instance(row, r) for _, row in self.X_test.iterrows()]

        cov = np.sum(cover_vec) / self.n
        cov_test = np.sum(cover_vec_test) / self.n

        return cov, cov_test

    def rule_accuracy(self, r):
        """
        Calculate rule accuracy
        :param r: rule r
        :return: accuracy of the rule
        """
        cover_vec = [self.rule_cover_instance(row, r) for _, row in self.X.iterrows()]
        class_vec = self.X[self.target_name].tolist() == np.repeat(float(r[-1]), self.n)
        cover_vec_class = [a and b for a, b in zip(cover_vec, class_vec)]

        cover_vec_test = [self.rule_cover_instance(row, r) for _, row in self.X_test.iterrows()]
        class_vec_test = self.X_test[self.target_name].tolist() == np.repeat(float(r[-1]), self.X_test.shape[0])
        cover_vec_class_test = [a and b for a, b in zip(cover_vec_test, class_vec_test)]

        acc = np.sum(cover_vec_class) / np.sum(cover_vec)
        acc_test = np.sum(cover_vec_class_test) / np.sum(cover_vec_test)

        return acc, acc_test

    def rule_recall(self, r):
        """
        Calculate rule recall
        :param r: rule r
        :return: recall of the rule
        """
        cover_vec = [self.rule_cover_instance(row, r) for _, row in self.X.iterrows()]
        class_vec = self.X[self.target_name].tolist() == np.repeat(float(r[-1]), self.n)
        cover_vec_class = [a and b for a, b in zip(cover_vec, class_vec)]

        cover_vec_test = [self.rule_cover_instance(row, r) for _, row in self.X_test.iterrows()]
        class_vec_test = self.X_test[self.target_name].tolist() == np.repeat(float(r[-1]), self.X_test.shape[0])
        cover_vec_class_test = [a and b for a, b in zip(cover_vec_test, class_vec_test)]

        num_instances = len(self.X[self.X[self.target_name] == float(r[-1])])
        num_instances_test = len(self.X_test[self.X_test[self.target_name] == float(r[-1])])

        recall = np.sum(cover_vec_class) / num_instances
        recall_test = np.sum(cover_vec_class_test) / num_instances_test

        return recall, recall_test

    def evaluate(self):
        """
        Evaluate the performance of the test data set with the train and test accuracy and the rule accuracy and coverage
        :return:
        """
        rs = [x for x in self.set_rules if 'None' not in x]
        # Accuracy
        pred = self.predict(self.X)
        acc = sum(self.X[self.target_name] == pred) / len(pred)

        pred_test = self.predict(self.X_test)
        acc_test = sum(self.X_test[self.target_name] == pred_test) / len(pred_test)

        # Rule coverage
        vec_cov = [self.rule_coverage(r)[0] for r in rs]
        vec_cov_test = [self.rule_coverage(r)[1] for r in rs]

        # Rule accuracy
        vec_acc = [self.rule_accuracy(r)[0] for r in rs]
        vec_acc_test = [self.rule_accuracy(r)[1] for r in rs]

        # Rule recall
        vec_rec = [self.rule_recall(r)[0] for r in rs]
        vec_rec_test = [self.rule_recall(r)[1] for r in rs]

        file1 = open("results/evaluation_" + self.name + ".txt", "w")
        # Write metrics
        file1.write("#-------\n")
        file1.write("Metrics\n")
        file1.write("#-------\n")
        file1.write("Train accuracy: " + str(acc) + "\n")
        file1.write("Test accuracy: " + str(acc_test) + "\n")
        file1.write("\n")
        file1.write("Rules accuracy train: " + str(vec_acc) + "\n")
        file1.write("Rules accuracy test: " + str(vec_acc_test) + "\n")
        file1.write("\n")
        file1.write("Rules coverage train: " + str(vec_cov) + "\n")
        file1.write("Rules coverage test: " + str(vec_cov_test) + "\n")

        # Write rules
        file1.write("\n#-------\n")
        file1.write("Rules\n")
        file1.write("#-------\n")
        rules_txt = self.print_rules()
        for r in rules_txt:
            file1.writelines(r)
            file1.write("\n")
        file1.close()

        # Create df with rules
        dict_df = {'Rules': rules_txt, 'Train coverage': vec_cov, 'Train accuracy': vec_acc, 'Train recall': vec_rec}
        df = pd.DataFrame(data=dict_df)
        df.sort_values(by=['Train recall', 'Train coverage', 'Train accuracy'], inplace=True, ascending=False)
        df.to_csv("results/rules_" + self.name + ".csv")

        dict_df = {'Rules': rules_txt, 'Test coverage': vec_cov_test, 'Test accuracy': vec_acc_test, 'Test recall': vec_rec_test}
        df = pd.DataFrame(data=dict_df)
        df.sort_values(by=['Test recall', 'Test coverage', 'Test accuracy'], inplace=True, ascending=False)
        df.to_csv("results/rules_" + self.name + "_test.csv")


