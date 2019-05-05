import numpy as np
from pandas import read_csv as read
import json

class Tree:
    def __init__(self):
        path = "bank.csv"
        dataset = read(path, delimiter=";")
        # %%
        dataset['y'] = dataset['y'].map({'no': 0, 'yes': 1})
        dataset['job'] = dataset['job'].map({'admin.': 0, 'unknown': 1,
                                             'unemployed': 2,
                                             'management': 3,
                                             'housemaid': 4,
                                             'entrepreneur': 5,
                                             'student': 6,
                                             'blue-collar': 7,
                                             'self-employed': 8,
                                             'retired': 9,
                                             'technician': 10,
                                             'services': 11})

        dataset['marital'] = dataset['marital'].map({'married': 0,
                                                     'divorced': 1,
                                                     'single': 2})

        dataset['education'] = dataset['education'].map({'unknown': 0,
                                                         'primary': 1,
                                                         'secondary': 2,
                                                         'tertiary': 3})

        dataset['contact'] = dataset['contact'].map({'unknown': 0,
                                                     'telephone': 1,
                                                     'cellular': 2})

        dataset['poutcome'] = dataset['poutcome'].map({'unknown': 0,
                                                       'other': 1,
                                                       'failure': 2,
                                                       'success': 3})

        dataset['default'] = dataset['default'].map({'no': 0, 'yes': 1})

        dataset['housing'] = dataset['housing'].map({'no': 0, 'yes': 1})

        dataset['loan'] = dataset['loan'].map({'no': 0, 'yes': 1})

        dataset = dataset.drop('month', 1)

        self.dataset = dataset

    @staticmethod
    def entropy1(target_col):
        """
        Calculate the entropy of a dataset.
        The only parameter of this function is the target_col parameter which specifies the target column
        """
        elements, counts = np.unique(target_col, return_counts=True)
        entropy = np.sum(
            [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
        return entropy

    def info_gain(self, data, split_attribute_name, target_name="class"):
        """
        Calculate the information gain of a dataset. This function takes three parameters:
        1. data = The dataset for whose feature the IG should be calculated
        2. split_attribute_name = the name of the feature for which the information gain should be calculated
        3. target_name = the name of the target feature. The default for this example is "class"
        """
        # Calculate the entropy of the total dataset
        total_entropy = self.entropy1(data[target_name])

        # Calculate the entropy of the dataset

        # Calculate the values and the corresponding counts for the split attribute
        vals, counts = np.unique(data[split_attribute_name], return_counts=True)

        # Calculate the weighted entropy
        weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * self.entropy1(
            data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])

        # Calculate the information gain
        information_gain = total_entropy - weighted_entropy

        return information_gain

    def ID3(self, data, originaldata, features, target_attribute_name="class", parent_node_class=None):
        """
        ID3 Algorithm: This function takes five paramters:
        1. data = the data for which the ID3 algorithm should be run --> In the first run this equals the total dataset

        2. originaldata = This is the original dataset needed to calculate the mode target feature value of the original dataset
        in the case the dataset delivered by the first parameter is empty
        3. features = the feature space of the dataset . This is needed for the recursive call since during the tree growing process
        we have to remove features from our dataset --> Splitting at each node
        4. target_attribute_name = the name of the target attribute
        5. parent_node_class = This is the value or class of the mode target feature value of the parent node for a specific node. This is
        also needed for the recursive call since if the splitting leads to a situation that there are no more features left in the feature
        space, we want to return the mode target feature value of the direct parent node.
        """
        # Define the stopping criteria --> If one of this is satisfied, we want to return a leaf node#

        # If all target_values have the same value, return this value
        if len(np.unique(data[target_attribute_name])) <= 1:
            return np.unique(data[target_attribute_name])[0]

        # If the dataset is empty, return the mode target feature value in the original dataset
        elif len(data) == 0:
            return np.unique(originaldata[target_attribute_name])[
                np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

        # If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
        # the direct parent node is that node which has called the current run of the ID3 algorithm and hence
        # the mode target feature value is stored in the parent_node_class variable.

        elif len(features) == 0:
            return parent_node_class

        # If none of the above holds true, grow the tree!

        else:
            # Set the default value for this node --> The mode target feature value of the current node
            parent_node_class = np.unique(data[target_attribute_name])[
                np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

            # Select the feature which best splits the dataset
            item_values = [self.info_gain(data, feature, target_attribute_name) for feature in
                           features]  # Return the information gain values for the features in the dataset
            best_feature_index = np.argmax(item_values)
            best_feature = features[best_feature_index - 1]

            # Create the tree structure.
            # The root gets the name of the feature (best_feature) with the maximum information
            # gain in the first run

            tree = {best_feature: {}}

            # Remove the feature with the best inforamtion gain from the feature space
            features = [i for i in features if i != best_feature]

            # Grow a branch under the root node for each possible value of the root node feature
            better_predict = {}
            for value in np.unique(data[best_feature]):
                value = value

                # Split the dataset along the value of the feature with
                # the largest information gain and therwith create sub_datasets

                sub_data_min = data.where(data[best_feature] <= value).dropna()
                sub_data_max = data.where(data[best_feature] > value).dropna()

                min_info_gain = self.info_gain(sub_data_min, best_feature, target_attribute_name)
                max_info_gain = self.info_gain(sub_data_max, best_feature, target_attribute_name)

                better_predict[value] = {'min': ''}
                better_predict[value] = {'max': ''}

                better_predict[value]['min'] = min_info_gain
                better_predict[value]['max'] = max_info_gain

            q_j_t = self.find_better_predict(better_predict)

            for key, val in q_j_t.items():
                print(key)
                quit()
                # sub_data_min = data.where(data[best_feature] <= key).dropna()
                # sub_data_max = data.where(data[best_feature] > key).dropna()
                #
                # # Call the ID3 algorithm for each of those sub_datasets with
                # # the new parameters --> Here the recursion comes in!
                # subtree_left = self.ID3(sub_data_min, data, features, target_attribute_name, parent_node_class)
                # #subtree_right = self.ID3(sub_data_max, data, features, target_attribute_name, parent_node_class)
                #
                # # Add the sub tree, grown from the sub_dataset to the tree under the root node
                # #tree[best_feature][key] = {'left': ''}
                # #tree[best_feature][value] = {'right': ''}
                #
                # tree[best_feature][key] = subtree_left
                # #tree[best_feature][value]['right'] = subtree_right

            return tree

    def find_better_predict(self, better_predicts):
        result = []

        if len(better_predicts) <= 1:
            return better_predicts
        if len(better_predicts) == 0:
            print('Null')
            quit()

        for key, value in better_predicts.items():
            for key_inner, val in better_predicts.items():
                if value['max'] > val['max'] and value['min'] > val['min']:
                    result = np.append(result, key)
                elif value['max'] > val['max'] and (value['min'] + 0.001) > val['min']:
                    result = np.append(result, key)
                else:
                    continue

        delete = [key for key in better_predicts if key not in np.unique(result)]

        for key in delete:
            del better_predicts[key]

        return self.find_better_predict(better_predicts)


if __name__ == '__main__':
    tree_class = Tree()
    data = tree_class.dataset
    tree = tree_class.ID3(data=data, originaldata=data, features=list(data.columns.values), target_attribute_name='y')
    print(tree)
