# getting started
# python classes

#class Myclass:
#    x = 5

#jade = Myclass()
#print(jade.x)

#-------------------------

#class Person:
#    def __init__(self, name, age):
#        self.name = name
#        self.age = age

#p1 = Person("savana", 23)
#print(p1.name)
#print(p1.age)

#print(p1) # without the string function

#class Person:
#    def __init__(self,name, age):
#        self.name = name
#        self.age = age

#    def __str__(self):
        # make sure to not include the spaces.
#        return f"{self.name} is {self.age}"

#toyou = Person("Joh", 34)
#print(toyou)

# so without the string function, it will not print out the name or age, but rather print out the location of the function.
# with the string function it prints the location.
#-------------------------

#class People:
#  def __init__(mysillyobject, name, age):
#    mysillyobject.name = name
#    mysillyobject.age = age

#  def myfunc(abc):
#    print("Hello my name is " + abc.name)

#errs = People("John", 36)
#errs.myfunc()
#--------------------------

#class Sebastian:
#    def __init__(self, tobe, people):
#        self.tobe = tobe
#        self.people = people

#    def tellme(self):
#        print("call me {}".format(self.people))

#    def __str__(self):
#        return f"{self.tobe} is {self.people}"

#    def calculate(self, x, y):
#        return x + y + self.people


#selestian = Sebastian("james", 34)
#selestian.tellme()
#print(selestian)
#print(selestian.calculate(2,3))

#--------------------------------------------------   Linear regression here.

import tensorflow as tf
print(tf.__version__) # check the version (should be 2.x+)

import datetime
print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")

import numpy as np
import matplotlib.pyplot as plt

# Create features (using tensors)
X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels (using tensors)
y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Visualize it
plt.scatter(X, y)

# Set random seed

# Create a model using the Sequential API
# Set random seed
tf.random.set_seed(42)

# Create a model (same as above)
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Compile model (same as above)
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.legacy.SGD(),
              metrics=["mae"])

# Fit model (this time we'll train for longer)
model.fit(tf.expand_dims(X, axis=-1), y, epochs=100) # train for 100 epochs not 10
