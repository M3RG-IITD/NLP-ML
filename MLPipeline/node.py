class node():
    def __init__(self, name="Node Name"):
        self.pipeline = None
        self.name = name
        self.isitnode = True
        self.steps = []

    def __str__(self,):
        return "{}".format(self.name)

    def __call__(self, *args, **kwargs):
        print("Node name: ",self.name)
        return self.run(*args,**kwargs)

    def run(self, X, order, *args, **kwargs):
        filename = "{}_Pipeline_{}_Node_{}".format(order, self.pipeline, self.name)
        for step in self.steps:
            X = step(X, filename, *args, **kwargs)
        return X

class connector(node):
    def __init__(self, name="Connector"):
        super().__init__(name)
        self.steps = [self.do]
    def do(self, indata):
        return indata
