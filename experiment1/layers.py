import ivy

class BaseDense:
    def __init__(self, input_size, output_size, batch_size):
        self.i_size = input_size
        self.o_size = output_size
        self.b_size = batch_size

        self.params = ivy.Container({
            "w": ivy.random_normal(shape=(self.i_size, self.o_size)),
            "b": ivy.zeros(shape=(self.o_size))
            })
        

        self.inputs = ivy.empty(shape=(self.b_size, self.i_size))
        self.out = ivy.empty(shape=(self.b_size, self.o_size))

        self.grads = ivy.Container({
            "w_g": ivy.zeros(shape=(self.i_size, self.o_size)),
            "b_g": ivy.zeros(shape=(self.i_size, self.o_size))
            })
    
    def forward(self, x):
        x = ivy.matmal(x, self.params["w"])
        ivy.add(x, self.params["b"], out=self.out)
        return self.out
    
    def backward(self, grads):
        self.grads["b_g"][:] = ivy.mean(grads, axis=0)
        for i in range(len(self.b_size)):
            fn = lambda j,k: self.inputs[i][j]*grads[i][k]
            self.grads["w_g"][:] += fn(*ivy.indices(self.params["w"].shape, dtype=self.params["w"].dtype))
        self.grads["w_g"]/= self.b_size
        return ivy.matmul(self.params["w"].T, grads)

    def __call__(self, x):
        return self.forward(x)
    
    



class CustomBackDense:
    def __init__(self, input_size, output_size, batch_size):
        self.i_size = input_size
        self.o_size = output_size
        self.b_size = batch_size

        self.w = ivy.random_normal(shape=(self.i_size, self.o_size))
        self.b = ivy.random_normal(shape=(self.o_size))

        self.inputs = ivy.empty(shape=(self.b_size, self.i_size))
        self.out = ivy.empty(shape=(self.b_size, self.o_size))

        self.w_g = ivy.zeros(shape=(self.i_size, self.o_size))
        self.b_g = ivy.zeros(shape=(self.i_size, self.o_size))