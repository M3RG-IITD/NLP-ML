from .node import connector

class pipe():
    def __init__(self, nodes=None, data=None, name="Pipeline name", output="."):
        if nodes==None:
            self.nodes = []
        else:
            self.nodes = nodes
            for item in self.nodes:
                item.pipeline = self.name
        self.data = data
        self.name = name
        self.isitpipe = True
        self.output = output

    def __str__(self):
        return "\nNumber of nodes: {}\n".format(self.nodes.__len__()) + "\n".join(["Node {}: {}".format(ind, node.__str__()) for ind,node in enumerate(self.nodes)]) + "\n"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        temp = pipe(nodes = self.nodes + other.nodes)
        return temp

    def __iadd__(self, other):
        self.nodes += other.nodes
        return self

    def __call__(self, X=None):
        if type(X)!=type(None):
            self.data = X
        return self.run()

    def add(self,x):
        if "isitnode" in x.__dir__():
            x.pipeline = self.name
            self.nodes += [x]
        elif "isitpipe" in x.__dir__():
            self.nodes += x.nodes
        else:
            raise Exception("Input is not of type: Node.")

    def addn(self, xs):
        pipes = [pipe(name=f"Diverging pipe {i}") + x for i,x in enumerate(xs)]
        class node_(connector):
            def __init__(self, pipes, name="Diverge"):
                super().__init__(name)
                self.pipes = pipes
            def do(self, data):
                for p in self.pipes:
                    p(data)
        self.nodes += [node_(pipes)]

    def remove(self, *args, node_name=None):
        if len(args)>1:
            raise Exception("Take only 1 positional argument i.e. node index.")
        if node_name!=None:
            for ind, item in enumerate(self.nodes):
                if item.name==node_name:
                    self.nodes.pop(ind)
        else:
            ind, = args
            if ind+1>self.nodes.__len__():
                raise Exception("Index out of bounds. Total number of nodes are {}".format(self.nodes.__len__()))
            else:
                self.nodes.pop(ind)

    def insert_at(self, ind, x):
        if ind>self.nodes.__len__():
            raise Exception("Index out of bounds. Total number of nodes are {}".format(self.nodes.__len__()))
        else:
            if "isitnode" in x.__dir__():
                x.pipeline = self.name
                self.nodes.insert(ind, x)
            elif "isitpipe" in x.__dir__():
                self.nodes = self.nodes[:ind]+x.nodes+self.nodes[ind:]
            else:
                raise Exception("Input is not of type: Node.")

    def run(self,):
        print("\nPipeline {} is running".format(self.name))
        for ind, item in enumerate(self.nodes):
            print("\nNode {} running.".format(ind))
            self.data = item(self.data, self.output + "/" + str(100+ind)[1:])
        return self.data

    def cut(self,ind,left=True):
        if ind>self.nodes.__len__():
            raise Exception("Index out of bounds. Total number of nodes are {}".format(self.nodes.__len__()))
        else:
            if left:
                return pipe(nodes=self.nodes[:ind], name=self.name+"_1"), pipe(nodes=self.nodes[ind:], name=self.name+"_1")
            else:
                return pipe(nodes=self.nodes[:ind-1], name=self.name+"_1"), pipe(nodes=self.nodes[ind-1:], name=self.name+"_1")
