# 新的需求

Keras 需要版本 [1.0.2] `pip install keras==1.0.2`
Tensorflow-gpu 需要低版本，因为keras是低版本，所以tf的高版本会导致concatenate失败，因为参数位置发生了改变
`pip install tensorflow-gpu==0.12`

所以最后推荐还是用`virtualenv`吧

[VGG16](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing)下载