# TensorflowStudy
Tensorflow学习代码

## 1.Tensorflow.py  
## 2.Tensorflow_Session.py  
## 3.Tensorflow_Variable.py  
## 4.Tensorflow_placeholder.py  
## 5.Tensorflow_add_layer.py  
## 6.Tensorflow_plot_result.py（安装matplotlib图形界面）  
## 7.Tensorflow_Tensorboard.py  
Window环境，能生成文件，但是浏览器打开后没有数据出来，不知道为什么？  
>解决了，第一生成的文件目录中不能包含中文；第二可以使用相对路径和绝对路径tensorboard --logdir=logs，这样它就会找到logs这个目录，而这里的logs目录是在py文件中  

![image](https://raw.githubusercontent.com/DyncKathline/Blog/master/Tensorflow/q1_Tensorflow_Tensorboard.png)  
![image](https://raw.githubusercontent.com/DyncKathline/Blog/master/Tensorflow/q2_Tensorflow_Tensorboard.png)  
## 8.Tensorflow_Tensorboard1.py  
## 9.Tensorflow_Classification.py（要运行这个成功，步骤如下）  
http://yann.lecun.com/exdb/mnist/ 从这里可以下载到这四个文件  
train-images-idx3-ubyte.gz:  training set images (9912422 bytes)   
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)   
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)   
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)  
>在当前目录下新建一个文件夹MNIST_data, 把下载的上面四个文件放入这个文件夹内。  

![image](https://raw.githubusercontent.com/DyncKathline/Blog/master/Tensorflow/q1_Tensorflow_Classification.png)  
![image](https://raw.githubusercontent.com/DyncKathline/Blog/master/Tensorflow/q2_Tensorflow_Classification.png)  
## 10.Tensorflow_overfitting.py  
## 11.Tensorflow_conv2d.py  
## 12.Tensorflow_conv2d1.py  
这里的11、12理解可[参考](http://note.youdao.com/noteshare?id=81b58cad78609b24b3aa37eacf154f51&sub=529A3D63DFB24228A099925F207C3651)中的“卷积函数 tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)”这部分  
