import numpy as np

#tinh sigmoid 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#tham so x la mot so da duoc tinh sigmoid, tra ve dao ham 
def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNerwork:
    def __init__(self, input_size, hidden_size, num_hidden_layer, output_size):
        np.random.seed(42)  
        self.Wi = np.random.rand(input_size, hidden_size)
        self.Wo = np.random.rand(hidden_size, output_size)
        self.Wh = []
        for _ in range(num_hidden_layer - 1):
            self.Wh.append(np.random.rand(hidden_size, hidden_size))

        self.all_hidden_activations = [] #luu tru dau ra sau ham kich hoat cua hidden_neural
        self.all_hidden_Wsum = [] #luu tru tong trong so (truoc ham kich hoat) cua hidden_neural
        self.final_Wsum = None #dau ra cuoi cung sau ham kich hoat, day la input cua neural cuoi cung
        self.final_activation = None #dau ra cuoi cung cua mang (du doan), day la output cua neural cuoi cung

    def forward(self, inputs):
        #reset lai luu tru
        self.all_hidden_activations = [] 
        self.all_hidden_Wsum = [] 

        #dau vao 
        current_weight_sum = np.dot(inputs, self.Wi) #tinh tong trong so dau vao dau tien
        current_activation = sigmoid(current_weight_sum) #ham kich hoat 

        self.all_hidden_Wsum.append(current_weight_sum)
        self.all_hidden_activations.append(current_activation)

        for W in self.Wh:
            current_weight_sum = np.dot(current_activation, W) #dau vao cua neural tiep theo la output cua neural truoc
            current_activation = sigmoid(current_weight_sum) #tiep tuc thong qua ham kich hoat

            self.all_hidden_Wsum.append(current_weight_sum)
            self.all_hidden_activations.append(current_activation)
            
        #dau ra cuoi cung
        self.final_Wsum = np.dot(current_activation, self.Wo) #tong trong so cuoi cung (Weight out)
        self.final_activation = sigmoid(self.final_Wsum) #output cua neural 
        
    
    def backpropagation(self, inputs, targets):
     
        #tinh toan chenh lech giua muc tieu dau ra va muc tieu mong muon
        error_output = targets - self.final_activation 
        #tinh delta, (chenh lech loi) * (dao ham cua ham kich hoat)
        delta_output = error_output * sigmoid_derivative(self.final_activation)
        # print(f"Gradient: {delta_output}")
        #cap nhat trong so cho Wo (weight out)
        #self.all_hidden_activations[-1] la dau ra cua hidden_neural cuoi cung, tuc day la input ket hop voi Wo, input cua neural cuoi cung
        self.Wo += self.all_hidden_activations[-1].T.dot(delta_output)
        '''
        Cong thuc chung ta thuong thay la W_new = W_old - learning_rate * (gradient)
        Nhung o day cach chung ta cap nhat trong so lai la +
        Do la boi vi ban chat cua gradient descent la tim huong di trai nguoc lai huong cua dao ham tang, de di tim cuc tieu
        Trong truong hop nay chung ta da tinh toan error_output chi ra dung huong ma chung ta can di roi (tuc huong di dung de co the di ve cuc tieu)
        Neu nhu muon dung cong thuc cap nhat trong so W_new = W_old - learning_rate * (gradient)
        Chung ta phai tinh toan error_output nhu sau
        delta_output = (output - targets) * sigmoid_derivative(final_activation)
        self.Wo -= self.all_hidden_activations[-1].T.dot(delta_output)
        chi can dao vi tri output va targets la duoc

        thuc chat learning rate cua chung ta o day la 1 (ma nhan voi 1 thi ghi vao cung cha co tac dung gi)
        toi da thu thay lr = 0.1 hay 0.001, nhan ra voi bai toan don gian nay lr qua be thi no lai lam du doan sai di
        cho nen lr = 1 co ve la phu hop nhat voi bai toan XOR nay
        '''

        #tinh delta cho hidden_neural cuoi cung
        #cthuc: previous_delta * (next layer weights).T * dao ham cua ham kich hoat (sigmoid_derivative)
        delta_hidden = delta_output.dot(self.Wo.T) * sigmoid_derivative(self.all_hidden_activations[-1])
        # print(f"Gradient: {delta_hidden}")
        #bay gio da co delta cua hidden_neural cuoi cung, ta co the tu no de lan truyen nguoc lai cac hidden_neural truoc do
        #range(start, end, step), bat dau tu cuoi len dau, start tu phan tu cuoi cung - 1 (bat dau tu 0), end la -1 tai vi phan tu dau tien la 0, step -1 de no di lui
        for i in range(len(self.Wh) - 1, -1, -1):

            self.Wh[i] += self.all_hidden_activations[i].T.dot(delta_hidden)  #cap nhat trong so cho hidden_neural
            delta_hidden = delta_hidden.dot(self.Wh[i].T) * sigmoid_derivative(self.all_hidden_activations[i])  #cap nhap delta_hidden de tiep tuc su dung cho cac hidden_neural tiep theo
            # print(f"Gradient: {delta_hidden}")
        
        #tai day su dung inputs, tai vi inputs chinh la dau vao cua neural dau tien
        self.Wi += inputs.T.dot(delta_hidden)
        print(f"Gradient: {delta_hidden}")
        
    def train(self, inputs, targets, epochs=10000):
        for epoch in range(epochs):
            self.forward(inputs)  
            self.backpropagation(inputs, targets)
            if epoch % (epochs/10) == 0:
                loss = np.mean(np.square(targets - self.final_activation)) #mean squared error
                print(f"Epoch {epoch}, Loss: {loss:.4f}")



def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
    Y = np.array([[0], [1], [1], [0]])  
    soluong_hidden_layer = 2 #max la 3, neu de la 4 hoac to hon se xay ra hien tuong vanishing gradient dan toi du doan khong dung
    '''
    Gradient: [[ 2.51457867e-04  2.49592492e-04 -3.97582583e-04]
    [-4.30928209e-06 -3.57867748e-05  9.85362439e-06]
    [-4.06004613e-05 -3.04399613e-06  7.54324690e-06]
    [ 2.18802597e-05  1.82842456e-05 -5.15937142e-08]]
    Ben tren la gia tri gradient cua lop an cuoi cung khi backpropagation
    khi hidden layer = 2, cac gia tri gradient chu yeu nam trong khoang 10^-3 ~ 10^-5
    0 XOR 0 = 0.0092
    0 XOR 1 = 0.9919
    1 XOR 0 = 0.9918
    1 XOR 1 = 0.0057
    ket qua cho ra co the noi la chinh xac

    Nhung sau khi doi so luong hidden layer = 5 thi day la su khac biet
    Gradient: [[ 2.43619231e-06 -1.13919314e-07  1.17603320e-06]
    [-1.98926811e-06  1.71421189e-07 -1.02002735e-06]
    [-2.00646227e-06  1.40274533e-07 -8.53559297e-07]
    [ 1.48394099e-06 -1.59441360e-07  7.00048208e-07]]
    gradient nam trong khoang 10^-7, tuc no da nho hon gap 100 lan so voi hidden layer = 2
    dieu nay la vi doi voi mot bai toan qua don gian nhu nay
    nhung lai de so luong layer qua nhieu, sau cac chuoi bien doi toan hoc qua tung lop
    gradient da nho den muc khong the cap nhat trong so duoc nua
    hay noi cach khac gradient khong du lon de trong so thay doi
    day chinh la hien tuong vanishing gradient
    0 XOR 0 = 0.5000
    0 XOR 1 = 0.5000
    1 XOR 0 = 0.5000
    1 XOR 1 = 0.5000
    day la ket qua
    '''
    network = NeuralNerwork(input_size=2, hidden_size=3, num_hidden_layer=soluong_hidden_layer, output_size=1)
    network.train(X, Y, epochs=10000)

    Y_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    network.forward(Y_test)
    print("Pre cho test data:")
    for i in range(len(Y_test)):
        a,b = Y_test[i]
        print(f"{a} XOR {b} = {network.final_activation[i][0]:.4f}")

if __name__ == "__main__":
    main()



        




        
        

       