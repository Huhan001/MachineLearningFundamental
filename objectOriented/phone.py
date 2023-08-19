from item import Item
class phone(Item):
    def __init__(self,name, price, quantity, broke_phone):
        super().__init__(name, price,quantity)
        self.broken_phone = broke_phone
