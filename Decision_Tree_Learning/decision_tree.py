__author__ = "Md. Ahsan Ayub"
__email__ = "mayub42@tntech.edu"
__last_edited__ = 1602735655190

#!/usr/bin/env python3
#

# Importing libraries
import sys
import math
import pandas as pd
import numpy as np
from typing import Dict

# Global variables
tree_records = []
tree_records_dict = {}
tree_records_conditions = {}


class node:
    ''' This is the defination class for each node in the constructued decision tree.
    Each node contains all the necessary details, e.g., name of the attribute, information
    gain, etc., and pointer to point its parent node, child node, and siblings (this is to
    ensure we can effectively construct one parent node can carry multiple childs).
    
    Reference on how it works: http://cs360.cs.ua.edu/lectures/8%20Non-Binary%20Trees%20and%20Traversals.pdf '''
    
    def __init__(self):
        self.data = {}
        self.parent = None
        self.leftmostChild = None
        self.rightSibling = None
        
    def get_data(self) -> dict:
        return self.data
    
    def set_data(self, data: Dict[str, str]):
        self.data = data
    
    def get_parent(self):
        return self.parent
    
    def set_parent(self, node_obj):
        self.parent = node_obj
    
    def get_leftmostChild(self):
        return self.leftmostChild
    
    def set_leftmostChild(self, node_obj):
        self.leftmostChild = node_obj
    
    def get_rightSibling(self):
        return self.rightSibling
    
    def set_rightSibling(self, node_obj):
        self.rightSibling = node_obj


def get_data_from_parameter(file_names):
    ''' This method takes the files names passed to execute the program.
    The file names include train and test sets along with the list of attributes.
    It returns both training and testing instances in a dataframe and attribtes in
    dictionary for further tasks. '''
    
    # Retrieve the file names passed in arguments
    train_file_name = file_names[0]
    test_file_name = file_names[1]
    attr_file_name = file_names[2]
    
    ''' The dictionary to store the attributes
    The keys will be the name of the attribute
    Its all possible values will be stored in a list. '''
    attr_dict = {}
    
    # Buffer through the attribute file to populate the dictionary as described
    with open(attr_file_name, "rb") as f:
        try:
            for line in f:
                flag_counter = 1
                temp = []
                for word in line.split():
                    word = word.decode("utf-8")
                    if flag_counter == 1:
                        key = word    
                        flag_counter = 0
                    else:
                        temp.append(word)
                        
                attr_dict[key] = temp
        except:
            print("Could not parse the attribute file; cannot continue.")
            exit(1)
    
    # Populate the train set to a dataframe
    if train_file_name.split('.')[-1] == 'txt':
        train = pd.read_csv(train_file_name, sep=" ", header=None)
    elif train_file_name.split('.')[-1] == 'csv':
        train = pd.read_csv(train_file_name, header=None)
    elif train_file_name.split('.')[-1] == 'xls' or train_file_name.split('.')[-1] == 'xlsx':
        train = pd.read_excel(train_file_name, header=None)
    else:
        print("Could not parse the train file; cannot continue.")
        exit(1)
    
    # Populate the test set to a dataframe
    if test_file_name.split('.')[-1] == 'txt':
        test = pd.read_csv(test_file_name, sep=" ", header=None)
    elif test_file_name.split('.')[-1] == 'csv':
        test = pd.read_csv(train_file_name, header=None)
    elif test_file_name.split('.')[-1] == 'xls' or test_file_name.split('.')[-1] == 'xlsx':
        test = pd.read_excel(train_file_name, header=None)
    else:
        print("Could not parse the test file; cannot continue.")
        exit(1)
    
    # Conversation of all the values to string
    train = train.applymap(str)
    test = test.applymap(str)
    
    # Fetching the list of columns from attributes
    train.columns = test.columns = [key for key in attr_dict]

    return train, test, attr_dict


def compute_entropy(attr_index, examples):
    ''' This method computes the entropy of the given attribute related to the class.
    The mathematical computations and relevant resources can be found here -
    https://towardsdatascience.com/entropy-and-information-gain-b738ca8abd2a '''
    
    # Fetch only the columns required to calculate entropy with class
    examples = examples.iloc[:, [attr_index, len(examples.columns)-1]]
    examples.columns = ['T', 'X'] # Rename the columns
    names, counts = np.unique(examples['T'], return_counts=True)
    
    entropies = []
    
    #print(names, counts)
    for i in range(0,(len(names))):
        df = examples[examples['T'] == names[i]]
        entropies.append(counts[i] / np.sum(counts))
        classes = examples['X'].unique()
        p1 = df.X.eq(classes[0]).sum() / len(df)
        p2 = df.X.eq(classes[1]).sum() / len(df)
        P = [p1, p2]
        #print(P)
        ent = 0
        k = 0
        for p in P:
            k = k + 1
            if(p != 0):
                ent += - (p * math.log2(p))
            if(k == len(P)):
                entropies.append(ent)
    #print(entropies)
    
    ent = 0
    i = 0
    while i < len(entropies)-1:
        ent += entropies[i] * entropies[i+1]
        i = i + 2
    #print(ent)
    return ent
        
    
def compute_information_gain(attr_index, examples):
    ''' This method computes the information gain of the given attribute related to its entropy.
    The mathematical computations and relevant resources can be found here -
    https://towardsdatascience.com/entropy-and-information-gain-b738ca8abd2a '''    
    
    #print("=== Entropy === ")
    ent = compute_entropy(attr_index, examples)
    
    # Compute H(X)
    X = examples.iloc[:, len(examples.columns)-1]
    names, counts = np.unique(X, return_counts=True)
    h_X = 0
    for i in range(0,(len(names))):
        if counts[i] > 0:
            h_X += - ((counts[i] / np.sum(counts)) * math.log2(counts[i] / np.sum(counts)))
    #print("=== Information Gain === ")
    inf_gain = h_X - ent
    #print(inf_gain)
    #print("\n")
    return inf_gain
        

def Decision_Tree_Learning(examples, attributes, node_obj: node):
    ''' This is the main function that recursively iterate through the dataset to learn
    the decision rules with the help of entropy and information gain. The algorithm is
    provided in the Russell and Norvig’s book “Artificial Intelligence, A Modern Approach.”
    
    Reference:
    1. https://ocw.mit.edu/courses/sloan-school-of-management/15-097-prediction-machine-learning-and-statistics-spring-2012/lecture-notes/MIT15_097S12_lec08.pdf
    2. http://vlm1.uta.edu/~athitsos/courses/cse4309_fall2020/lectures/11a_decision_trees.pdf '''
    
    ''' Base conditions to stop the recursions '''
    if examples.empty: # no more dataframe
        #print("no more dataframe")
        return
    elif not attributes: # no more columns to perform split
        print("no more attributes")
        return
    else:
        ''' Compute the information gains and entropies to get the best attributes and perform
        the decision solits '''
        inf_gains = {}
        df_columns = examples.columns.tolist()
        df_columns = df_columns[:-1] # removing class entry
        for df_cols in df_columns:
            if df_cols in attributes:
                inf_gains[df_cols] = compute_information_gain(examples.columns.get_loc(df_cols), examples)
            else:
                continue
        
        max_info_gain = -1
        attr_name = ''
        for key in inf_gains:
            if inf_gains[key] > max_info_gain:
                max_info_gain = inf_gains[key]
                attr_name = key
        
        '''print(inf_gains)
        print("=== xxx ===")
        print(attr_name)
        print("=== xxx ===")'''
        
        ''' Creating decision node for the tree '''
        n = node() # Create the parent node
        node_data = {}
        node_data['attribute_name'] = attr_name
        node_data['information_gain'] = str(max_info_gain)
        
        if node_obj == None: # Root
            node_data['condition'] = 'root'
            n.set_parent(None)
            n.set_rightSibling(None)
            n.set_leftmostChild(None)
            n.set_data(node_data)
            tree_records.append(n)
            #print('root is created')
        else:
            n.set_parent(node_obj)
            node_data['condition'] = ''
            n.set_rightSibling(None)
            n.set_leftmostChild(None)
            n.set_data(node_data)
            tree_records.append(n)
            #print('%s is created' % attr_name)
        
        ''' Child nodes for the decision node of the tree '''
        df = examples.iloc[:, [examples.columns.get_loc(attr_name), len(examples.columns)-1]]
        df.columns = ['attr', 'class']
        names, counts = np.unique(df['attr'], return_counts=True)
        #print(names, counts)
        
        tree_records_conditions[attr_name] = np.unique(df['attr']).tolist()
        
        for i in range(0,(len(names))): # Work with the definative columns first
            dff = examples[examples[attr_name] == names[i]]
            node_data = {}
            if len(dff['class'].unique()) == 1:
                #print("%s is being deteleted" % str(names[i]))
                
                temp = node() # Create the temp nodes to link the current node
                node_data['condition'] = str(names[i])
                node_data['attribute_name'] = dff['class'].unique()[0]
                node_data['information_gain'] = None
                
                temp.set_rightSibling(None)
                temp.set_leftmostChild(None)
                temp.set_data(node_data)
                temp.set_parent(n)
                #print(temp.get_data())
                tree_records.append(temp)
                
                #print('%s is created' % node_data['attribute_name'])
                
                del temp
                examples = examples.drop(examples[examples[attr_name] == names[i]].index)
        
        ''' Removing the attribute from the consideration for the next recusive round '''
        if attr_name in attributes:
            attributes.remove(attr_name)
        Decision_Tree_Learning(examples, attributes, n)


def build_tree():
    ''' This is a supplementary method added to build the finer connections of the nodes created
    in the decision-tree-learning method above. This method connects all the sibling nodes in a
    layer as well as adds condition for the parent nodes that will help the traversal of the tree. '''
    
    # Create a dictory of parent nodes and its child nodes as its values for easier usage
    tree_records_dict[tree_records[0].get_data()['attribute_name']] = { 'node_obj' : tree_records[0], 'child' : []}
    
    # Adding condition for the parent nodes
    for n in tree_records[1:]:
        if n.get_data()['condition'] == '':
            tree_records_dict[n.get_data()['attribute_name']] = { 'node_obj' : n, 'child' : []}
        
        if n.get_parent():
            tree_records_dict[n.get_parent().get_data()['attribute_name']]['child'].append(n)

    # Adding connectations to all the sibling nodes
    decision_node = node()
    for key in tree_records_dict:
        conditions = set(tree_records_conditions[key])
        conditions_terminating_nodes = set()
        for child_node in tree_records_dict[key]['child']:
            if child_node.get_data()['condition'] != '':
                conditions_terminating_nodes.add(child_node.get_data()['condition'])
            else:
                decision_node = child_node
        if (conditions - conditions_terminating_nodes):
            node_data = decision_node.get_data()
            for item in (conditions - conditions_terminating_nodes):
                break
            node_data['condition'] = str(item)
            decision_node.set_data(node_data)
            
        tree_records_dict[key]['node_obj'].set_leftmostChild(tree_records_dict[key]['child'][0])
        for i in range(len(tree_records_dict[key]['child'])-1):
            tree_records_dict[key]['child'][i].set_rightSibling(tree_records_dict[key]['child'][i+1])


def print_decision_tree():    
    ''' As all the nodes are connected from the root, we traverse the constructed tree starting from
    the root. We preview the nodes and its connection with several pieces of information. '''
    
    node_obj = tree_records[0]    # Pointing at the root
    while(True):
        # Print operations
        if node_obj.get_parent():
            print("%s (Info Gain: %s and Condition from its parent, %s, is %s)" %
                  (node_obj.get_data()['attribute_name'], node_obj.get_data()['information_gain'],
                   node_obj.get_parent().get_data()['attribute_name'], node_obj.get_data()['condition']))
        else:
            print("%s (Info Gain: %s and Root)" % (node_obj.get_data()['attribute_name'], node_obj.get_data()['information_gain']))        
        
        # Iteration to the next node
        if node_obj.get_rightSibling():         # Check the siblings first
            node_obj = node_obj.get_rightSibling()
            continue
        
        if node_obj.get_leftmostChild():        # Check if the node has children
            node_obj = node_obj.get_leftmostChild()
            continue
        
        break    # Break from the infinite loop


def predict(dataset):
    ''' This method computes the accuracy of both train and test set.
    For each record in the set, the function traverses through the tree to
    find the terminating class result. At the end, it compares the predicted values
    with the class values provided in the dataset. '''
    
    y_class = dataset.iloc[:, -1].tolist()
    X = dataset.iloc[:, 0:-1]
    y_pred = []
    sample_condition = ''
    
    # Traversal of the tree to predict the class label
    for i in range(len(X)):
        df = X.loc[i]
        node_obj = tree_records[0]
        while(True):
            if node_obj.get_data()['attribute_name'] in [attributes for attributes in tree_records_dict]:
                curr_attribute = node_obj.get_data()['attribute_name']
                sample_condition = df[curr_attribute]
                #print(node_obj.get_data(), sample_condition)
            else:
                if sample_condition == node_obj.get_data()['condition']:
                #if sample_condition == node_obj.get_data()['attribute_name']:
                    #print(node_obj.get_data()['attribute_name'], node_obj.get_parent().get_data()['attribute_name'])
                    y_pred.append(node_obj.get_data()['attribute_name'])
                    break
            
            if node_obj.get_rightSibling():        # Check the siblings first
                node_obj = node_obj.get_rightSibling()
                continue
            
            if node_obj.get_leftmostChild():     # Check if the node has children
                node_obj = node_obj.get_leftmostChild()
                continue
            
            # This will be valid when the tree cannot assign a class label to the record
            y_pred.append('Default')
            break
        
    # Compute the successful predictions
    correct_predictions = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_class[i]:
            correct_predictions += 1
    
    # Return the accuracy score
    return correct_predictions / len(y_pred)
    


# Utility function
if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Missing train, test, and arguments file names.")
        print("Usuage: python3 decision_tree.py <train.txt> <test.txt> <attr.txt>")
        exit(1)
    
    train, test, attr_dict = get_data_from_parameter(sys.argv[1:])
    print("Train, test, and attribute files are read properly")
    print("Training Set Shape: %s" % str(train.shape))
    print("Testing Set Shape: %s" % str(test.shape))

    ''' Learn the decision rules from the training dataset with the help of algorithm provided
    in the Russell and Norvig’s book “Artificial Intelligence, A Modern Approach.” '''
    Decision_Tree_Learning(train, [key for key in attr_dict], None)
    build_tree()
    
    print("\n============ Decision Tree ============")
    print_decision_tree()
    
    print("\n============ Performance ============")
    print("Train Accuracy: {:.4f}".format(predict(train)))
    print("Test Accuracy: {:.4f}".format(predict(test)))