import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class Tree:
    def __init__(self):
        path = "bank.csv"
        dataset = pd.read_csv(path, delimiter=";")
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
            if len(data[target_attribute_name]) == 0:
                return parent_node_class
            else:
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
            best_feature = features[best_feature_index]

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
                # # Split the dataset along the value of the feature with
                # # the largest information gain and therwith create sub_datasets
                #
                # sub_data = data.where(data[best_feature] == value).dropna()
                #
                # subtree = self.ID3(sub_data, data, features, target_attribute_name, parent_node_class)
                #
                # # Add the sub tree, grown from the sub_dataset to the tree under the root node
                # tree[best_feature][value] = subtree

                sub_data_min = data.where(data[best_feature] <= value).dropna()
                sub_data_max = data.where(data[best_feature] > value).dropna()

                min_info_gain = self.info_gain(sub_data_min, best_feature, target_attribute_name)
                max_info_gain = self.info_gain(sub_data_max, best_feature, target_attribute_name)

                better_predict[value] = {'min': ''}
                better_predict[value] = {'max': ''}

                better_predict[value]['min'] = min_info_gain
                better_predict[value]['max'] = max_info_gain

            q_j_t = self.find_better_predict(better_predict)
            print(best_feature)
            for key, val in q_j_t.items():
                # Split the dataset along the value of the feature with
                # the largest information gain and therwith create sub_datasets

                sub_data_left = data.where(data[best_feature] <= key).dropna()
                sub_data_right = data.where(data[best_feature] > key).dropna()

                subtree_left = self.ID3(sub_data_left, data, features, target_attribute_name, parent_node_class)
                subtree_right = self.ID3(sub_data_right, data, features, target_attribute_name, parent_node_class)

                tree[best_feature][key] = {'left': ''}
                tree[best_feature][key] = {'right': ''}

                # Add the sub tree, grown from the sub_dataset to the tree under the root node
                tree[best_feature][key]['left'] = subtree_left
                tree[best_feature][key]['right'] = subtree_right

            return tree

    def find_better_predict(self, better_predicts):
        result = []
        for_iter = better_predicts
        while len(better_predicts) > 1:
            for value in list(better_predicts):
                for val in list(for_iter):
                    print(val)
                    if better_predicts[value]['max'] > for_iter[val]['max'] and \
                            better_predicts[value]['min'] > for_iter[val]['min']:
                        result = np.append(result, value)
                    elif better_predicts[value]['max'] > for_iter[val]['max'] and \
                            (better_predicts[value]['min'] + 0.001) > for_iter[val]['min']:
                        result = np.append(result, value)
                    else:
                        if better_predicts[value]['max'] == 0.0 and \
                                better_predicts[value]['min'] > for_iter[val]['min']:
                            result = np.append(result, value)
                        elif better_predicts[value]['min'] == 0.0 and \
                                better_predicts[value]['max'] > for_iter[val]['max']:
                            result = np.append(result, value)
                        else:
                            for_iter.pop(val)
                            continue
            print(for_iter)
            quit()
            delete = [key for key in better_predicts if key not in np.unique(result)]

            for key in delete:
                del better_predicts[key]
            print(len(better_predicts))
        quit('dd')
        return self.find_better_predict(better_predicts)

    def predict(self, query, tree, default=1):

        for key in list(query.keys()):
            if key in list(tree.keys()):

                try:
                    result = tree[key][query[key]]
                except:
                    return default

                result = tree[key][query[key]]

                if isinstance(result, dict):
                    return self.predict(query, result)
                else:
                    return result

    def test(self, data, tree, target_attribute_name="class"):
        # Create new query instances by simply removing the target feature column from the original dataset and
        # convert it to a dictionary
        queries = data.iloc[:, :-1].to_dict(orient="records")

        # Create a empty DataFrame in whose columns the prediction of the tree are stored
        predicted = pd.DataFrame(columns=["predicted"])

        # Calculate the prediction accuracy
        for i in range(len(data)):
            predicted.loc[i, "predicted"] = self.predict(queries[i], tree, 1.0)

        print('The prediction accuracy is: ', (np.sum(predicted["predicted"].values == data[target_attribute_name].values) /
                                               len(data)) * 100, '%')


if __name__ == '__main__':
    tree_class = Tree()
    data = tree_class.dataset
    #train, test = train_test_split(data[0:25], test_size=0.2)
    tree = tree_class.ID3(data=data[0:25], originaldata=data[0:25], features=data.columns[:-1], target_attribute_name='y')
    print(tree)
    #tree_class.test(data=test, tree=tree, target_attribute_name='y')

    # train_features = train.drop('y', 1)
    # train_targets = train['y']
    # tree_in_sklearn = DecisionTreeClassifier(criterion='entropy').fit(train_features, train_targets)
    #
    # test_features = test.drop('y', 1)
    # prediction = tree_in_sklearn.predict(test_features)
    # print("The prediction accuracy is: ", tree_in_sklearn.score(test_features, test['y']) * 100, "%")
