import csv
class Item:
    # class attribute
    pay_rate = 0.5
    all = []
    def __init__(self,name: str, price: float, quantity = 0):
        # run validation to recieved items
        assert price >= 0, f"price {price} is not greater than zero"
        assert quantity >= 0

        self.__name = name
        self.__price = price
        self.__quantity = quantity

        # append to all
        Item.all.append(self)

    @property
    def quantity(self):
       return self.__quantity

    @property
    def price(self):
        return self.__price

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        self.__name = value

    def calculate_total_price(self):
        # we do not need to have designated parameter
        return  self.__price * self.__quantity

    def apply_discount(self):
        return self.__price * self.pay_rate


    # decorators are used to chaneg the behavior of a class method====================================
    @classmethod
    def instantiate_from_csv(cls):
        with open('item.csv','r') as f:
            reader = csv.DictReader(f)
            items = list(reader)

        for item in items:
            Item(
                name = item.get('name'),
                price = float(item.get('price')),
                quantity = int(item.get('quantity'))
            )
    # decorators are used to chaneg the behavior of a class method====================================
    @staticmethod
    def is_integer(num):
        #we will count out the decimals that are point zero
        if isinstance(num, float):
            # count out the float that are decimal point zero
            return num.is_integer()
        elif isinstance(num, int):
            return True
        else:
            return False



    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}', {self.__price}, {self.__quantity})"
