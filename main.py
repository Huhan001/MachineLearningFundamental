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

class Sebastian:
    def __init__(self, tobe, people):
        self.tobe = tobe
        self.people = people

    def tellme(self):
        print("call me {}".format(self.people))

    def __str__(self):
        return f"{self.tobe} is {self.people}"

    def calculate(self, x, y):
        return x + y + self.people


selestian = Sebastian("james", 34)
selestian.tellme()
print(selestian)
print(selestian.calculate(2,3))


