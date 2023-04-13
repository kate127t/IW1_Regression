
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class myLinearRegression:
    def __init__(self, standardize=False, 
                 learning_rate=0.01, 
                 num_iters=1000,
                 tol=1e-4,
                 print_J=False):
        """Initialize Linear Regression.
        
        Args:
            standardize (bool): Whether to standardize the data.
            learning_rate (float): Learning rate for gradient descent.
            max_iter (int): Maximum number of iterations for gradient descent.
            tol (float): Tolerance for gradient descent.
            print_J (bool): Whether to print cost at each 100th iteration.
        """
        self.standardize = standardize
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.tol = tol
        self.print_J = print_J

    def normalize(self,X):
        '''
        Нормалізує датасет з характеристиками
        
        Параметри:
        X - набір характеристик
        
        Результат:
        X_new - набір нормалізованих характеристик, (X-mean)/std
        mean - вектор середніх значень характеристик
        std - вектор стандартних девіацій характеристик
        '''
        mean = X.mean(axis = 0)
        std = X.std(axis = 0)
        X_new  = np.divide(np.subtract(X,mean),std)
        return X_new, mean, std
    
    def prepare_X(self,X):
        '''
        Формує датасет з рисами, де першою колонкою буде колонка з одиницями.
        
        Параметри:
        X - вхідний датасет з прикладами, по одному в рядок. Кожна риса - відповідна колонка.
        
        Результат:
        X_new - датасет, який складається з колонки одиниць, а решта колонок з X    
        '''
        m = X.shape[0]
        ones = np.ones((m, 1))
        X_new = np.array(X[:])
        X_new = np.column_stack((ones, X_new))

        return X_new
    
    def hypothesis(self,X, theta):
        '''
        Обчислює значення передбачуваної величини для вхідної матриці X та вектора параметрів thetha.
        
        Параметри:
        X - матриця з рисами. Перша колонка - одиниці. Друга - дані риси.
        thetha - вектор параметрів: [thetha0, thetha1]
        
        Результат:
        Матриця значень шуканої величини для прикладів з X
        '''
        h_theta = np.dot(X, np.transpose(theta))
        
        return h_theta
    
    def cost_function(self,X, y, theta):
        '''
        Функція для обчислення штрафної функції J.
        
        Параметри:
        X - тренувальний датасет. 0 - колонка з одиниць, далі - реальні риси
        y - точні значення передбачуваної величини
        thethe - вектор параметрів регресії
        
        Результат:
        Дійсне число - значення штрафної функції для набору прикладів X та параметрів thetha
        '''
        m = X.shape[0]
        if m == 0:
            return None
        
        J = 1/(2*m) * (np.sum(np.power (self.hypothesis(X,theta) - y,2),axis = 0))

        return J
    
    def derivative(self,X, y, theta):
        m = X.shape[0]
        '''
        Функція для обчислення похідних штрафної функції J по thetha.
        
        Параметри:
        X - тренувальний датасет. 0 - колонка з одиниць, далі - реальні риси
        y - точні значення передбачуваної величини
        thetha - вектор параметрів регресії
        
        Результат:
        Вектор похідних d_thetha
        '''
        
        d_theta = 1/(m) * (np.dot ( np.transpose(X),(self.hypothesis(X,theta) - y)))

        return d_theta
    
    def gradient_descent(self,X, y, theta):
        '''
        Функція, що реалізує градієнтний спуск для метода лінійної регресії.
        
        Параметри:
        X - тренувальний датасет. 0 - колонка з одиниць, далі - реальні риси
        y - точні значення передбачуваної величини
        thetha - вектор початкових параметрів регресії
        alpha - швидкість навчання
        num_iters - кількість ітерацій
        print_J - виведення штрафної функції на екран після кожної ітерації
        
        Результат:
        theta - оптимальні значення параметрів регресії
        J_history - масив історичних значень штрафної функції після кожної ітерації
        
        
        1) J_i (theta_0, theta_1)
        2)  theta_0 = theta_0 - alpha*dtheta_0
            theta_1 = theta_1 - alpha*dtheta_1
            |J_i-J_{i-1}| < eps || num_iters>10000000000 -> break
        3) goto 1
        '''
        #X = self.prepare_X(X)
        m = X.shape[0]
        J_history = []
        J = self.cost_function(X, y, theta)
        if self.print_J == True:
            print(J)
        J_history.append(J)
        for i in range(self.num_iters):
            delta = self.derivative(X, y, theta)
            theta = np.subtract(theta,np.dot(self.learning_rate,delta)) 
            J = self.cost_function(X, y, theta)
            
            if self.print_J == True:
                print(J)
            J_history.append(J)
        return theta, J_history
    
    def fit(self, X, y):
        """Fit the model.

        Args:
            X (array): Data.
            y (array): Target."""
        if self.standardize:
            X_new, self.mean, self.std = self.normalize(X)
        X_new = self.prepare_X(X)

        self.theta = np.zeros(X_new.shape[1])
        self.theta, self.costs = self.gradient_descent(X_new, y, self.theta)

    def predict(self, X):
        """Predict the target.

        Args:
            X (array): Data.

        Returns:
            array: Predicted target."""
        if self.standardize:
            X_new = np.divide(np.subtract(X,self.mean),self.std)
        X_new = self.prepare_X(X)
        y_pred = self.hypothesis(X_new,self.theta)
        
        return y_pred
