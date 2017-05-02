import tensorflow as tf

hello_constant = tf.constant("Hello world")

#텐서 플로우는 기본 변수 저장을 tensor 단위로 한다.
#tf.constant ()에 의해 리턴 된 텐서는 텐서의 값이 변하지 않기 때문에 상수 텐서라 부른다.

# A is a 0-dimensional int32 tensor
A = tf.constant(1234)
# B is a 1-dimensional int32 tensor
B = tf.constant([123,456,789])
 # C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])

with tf.Session() as sess: #세션 메서드로 세션 인스턴스 sess를 만든다.
    #세션은 그래프를 실행하기 위한 환경이다.
    #세션은 원격 컴퓨터를 포함하여 GPU 및  CPU에 작업을 할당한다.

    #with as 구문은 어떤 블럭에 진입하고 나올 때 지정된 객체(context manager)로 하여금 그 시작과 끝에서 어떤 처리를 하도록 할 때 사용한다.
    #파일이나 DB 혹은 네트워크 연결을 열어서 작업하던 중에 예외가 발생하였을 때에 안전하게 리소스 처리를 할 수 있는 로직을 깔끔하게 처리할 수 있다.
    #파일 입출력 시에 try catch finally 등 대신 쓸 수 있다.
    output = sess.run(hello_constant) #세션 메서드 run으로 텐서를 평가하고 결과값을 반환한다.
    print(output)





################################ input ############################################
x = tf.placeholder(tf.string)
#모델 입력 시 변경되지 않을 데이처를 입력하고자 할 때는 placeholder를 사용하면 된다.
#상수(constant)처럼 바로 값을 입력할 수 없다.
#tf.placeholder()는 tf.session.run() 함수에 전달 된 데이터에서 값을 가져 오는 텐서를 반환하므로 세션이 실행되기 전에 입력을 바로 설정할 수 있다.
#자료형 뒤에 shape를 지정해 줄 수 도 있다. https://www.tensorflow.org/api_docs/python/tf/placeholder

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})
    #run()에서 feed_dict 매개 변수를 사용해서 텐서를 설정해 준다.
    #(변수 텐서(placeholder), feed_dict = 매개변수 설정(딕셔너리))
    #여기서는 텐서 x가 문자열 "Hello, world"로 설정된다.
    print(output)

x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
    #feed_dict를 사용하여 하나 이상의 텐서를 설정할 수도 있다.
    print(output)





################################ TensorFlow Math ################################
x = tf.add(5, 2)  # 7
x = tf.subtract(10, 4) # 6
y = tf.multiply(2, 5)  # 10

#tf.subtract(tf.constant(2.0),tf.constant(1))
#Fails with ValueError: Tensor conversion requested dtype float32 for Tensor with dtype int32:
#변수의 단위를 맞춰줘야 한다.

tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))   # 1

x = tf.constant(10)
y = tf.constant(2)

z = tf.subtract(tf.divide(x, y), tf.cast(tf.constant(1), tf.float64)) #x/y - 1





################################ Weights and Bias in TensorFlow ################################
#tf.placeholder()는 입력 데이터를 만들 때 주로 사용한다. (실제 훈련 예제를 제공하는 변수) - 초기값을 지정할 필요 없다. (모델 입력시 변경되지 않을 데이터)
#tf.Variable()은 데이터의 상태를 저장할 때 주로 사용한다. (가중치나 편향 등의 학습 가능한 변수) - 초기값을 지정해야 한다. (학습 되는 데이터)
#http://stackoverflow.com/questions/36693740/whats-the-difference-between-tf-placeholder-and-tf-variable

x = tf.Variable(5) #Variable()은 수정할 수 있는 텐서를 생성한다.
init = tf.global_variables_initializer() #global_variables_initializer() 모든 가변 텐서의 상태 초기화하는 메서드
with tf.Session() as sess:
    sess.run(init)
    #tf.global_variables_initializer()는 모든 TensorFlow 변수를 그래프에서 초기화하는 연산을 반환한다.
    #세션에서 작업을 호출 해 모든 변수를 초기화한다.
    #tf.Variable 클래스를 사용하면 가중치와 편향의 초기 값을 입력해야 한다.

n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels))) #tf.truncated_normal() 정규 분포 내에서 임의의 값을 생성한다.
#tf.truncated_normal()는 평균값으로부터 2 표준 편차를 넘지 않는 정규 분포에서 무작위 값을 갖는 텐서를 반환한다.
#정규 분포에서 무작위 수로 가중치를 초기화하는 것이 좋다.

bias = tf.Variable(tf.zeros(n_labels))
#가중치를 설정하면 학습 모델이 정체되는 것을 막을 수 있으므로 편향까지 무작위로 추출 할 필요는 없다. 따라서 편향을 0으로 설정하기도 한다.





################################ One Hot Encoding ################################
import numpy as np
from sklearn import preprocessing

# Example labels
labels = np.array([1,5,3,2,1,4,2,1,3])

# Create the encoder
lb = preprocessing.LabelBinarizer() #sklearn의 preprocessing을 통해 쉽게 one-hot encoding을 구현할 수 있다.

# Here the encoder finds the classes and assigns one-hot vectors
lb.fit(labels)

# And finally, transform the labels into one-hot encoded vectors
lb.transform(labels)
 #    array([[1, 0, 0, 0, 0],
 #           [0, 0, 0, 0, 1],
 #           [0, 0, 1, 0, 0],
 #           [0, 1, 0, 0, 0],
 #           [1, 0, 0, 0, 0],
 #           [0, 0, 0, 1, 0],
 #           [0, 1, 0, 0, 0],
 #           [1, 0, 0, 0, 0],
 #           [0, 0, 1, 0, 0]])





################################ Cross Entropy ################################
x = tf.reduce_sum([1, 2, 3, 4, 5])  # 15
#reduce_sum은 배열의 수를 더해서 반환

# 'x' is [[1, 1, 1]
#         [1, 1, 1]]
x = [[1, 1, 1], [1, 1, 1]]
tf.reduce_sum(x) # 6 #축을 설정하지 않으면 모든 원소를 더한다.
tf.reduce_sum(x, 0) # [2, 2, 2] #축 방향만 더해서 반환
tf.reduce_sum(x, 1) # [3, 3] #축 방향만 더해서 반환
tf.reduce_sum(x, 1, keep_dims=True) # [[3], [3]]
tf.reduce_sum(x, [0, 1]) # 6

# x = tf.log(100)  # 4.60517
#log는 자연로그를 취한다.

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)
cross_entropy = -tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))
#***** 크로스 엔트로피 오차는 원 핫 인코딩에서 해당 값만을 -ln(x) 한 것. *****
#tf.multiply(one_hot, tf.log(softmax))를 하면 정답 레이블만 값을 가지게 된다.
#cross_entropy = -tf.log(tf.reduce_sum(tf.multiply(one_hot, softmax)))도 같다.

with tf.Session() as sess:
    print(sess.run(cross_entropy, feed_dict={softmax: softmax_data, one_hot: one_hot_data}))





################################ Mini-batch ################################
import math
from pprint import pprint

def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    output_batches = []

    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]] #end_i가 범위를 넘어가면 마지막까지만 출력
        output_batches.append(batch)

    return output_batches

# 4 Samples of features
example_features = [
    ['F11','F12','F13','F14'],
    ['F21','F22','F23','F24'],
    ['F31','F32','F33','F34'],
    ['F41','F42','F43','F44']]
# 4 Samples of labels
example_labels = [
    ['L11','L12'],
    ['L21','L22'],
    ['L31','L32'],
    ['L41','L42']]

# PPrint prints data structures like 2d arrays, so they are easier to read
pprint(batches(3, example_features, example_labels))
