import numpy as np
import math


class Core:

    def start(self):
        data = np.array([[0, 0, 0, 1, 1, 0, 1, 1, 0],
                         [19, 20, 18, 25, 30, 20, 21, 45, 28],
                         [1500, 1000, 1000, 10000, 20000, 2700, 800, 10000, 2000]])
        ages = data[1]
        entropy_0 = self.entropy(data[0])
        self.information_gain(entropy_0, data)
        """
        Делим по критерию возраста
        """
        # for value in data:
        #     print(value)

    @staticmethod
    def entropy(array):
        """
        Считаем энтропию

        :return:
        """
        all_experiments = len(array)

        # Уникальные значения а массиве
        uniq_array = Core.uniq_elements_in_array(array)

        entropy_res = 0
        for item in uniq_array:
            entropy_res = entropy_res + Core.find_log(Core.count_element(array, item)/all_experiments)

        return entropy_res

    @staticmethod
    def uniq_elements_in_array(array):
        uniq_array = []

        for value in array:
            if value not in uniq_array:
                uniq_array.append(value)

        return uniq_array

    @staticmethod
    def find_log(p_i):
        """
        :param p_i: - вероятность события
        :return:
        """
        return - (p_i * math.log2(p_i))

    @staticmethod
    def count_element(array, point):
        """
        :param array:
        :param point:
        :return:
        """
        counter = 0

        for el in array:
            if el == point:
                counter = counter + 1

        return counter

    @staticmethod
    def information_gain(entropy_0, array):
        for value in Core.uniq_elements_in_array(array[1]):
            p_i = Core.count_element(array[1], value) / len(array[1])
            if p_i > (1 / len(array[1])):
                arr_index = np.where(array[1] == value)
                print(arr_index[0][1])


if __name__ == '__main__':
    Core().start()
