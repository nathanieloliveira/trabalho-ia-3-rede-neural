from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adadelta
import numpy as np

import csv


input_dimension = 32


def row_count(filename):
    with open(filename) as in_file:
        return sum(1 for _ in in_file)


def string_to_float_array(string, size):
    x = np.zeros(size)
    array = string.encode('utf-8')
    for i, b in enumerate(array):
        if i >= size:
            break
        x[i] = float(b)
    return x


def process_line(line):
    insc = line['NU_INSCRICAO']
    city = line['NO_MUNICIPIO_RESIDENCIA']
    c = str(insc) + str(city)
    x = string_to_float_array(c, input_dimension)

    grade = str(line['NU_NOTA_MT'])
    y = 0.0
    if len(grade) > 0:
        y = float(grade)
    return x, y


def generator(batch_size):
    while 1:
        filename = 'microdados_enem_2016_coma.csv'

        rows = 8627368
        file = open(filename)
        enem = csv.DictReader(file)

        inputs = np.zeros((batch_size, input_dimension))
        labels = np.zeros((batch_size, 1))

        read = 0
        while (read + batch_size) < rows:
            for i in range(batch_size):
                line = next(enem)
                read += 1
                x, y = process_line(line)
                inputs[i] = x
                labels[i] = y
            yield inputs, labels
        file.close()


model = Sequential()
model.add(Dense(32, activation='relu', input_dim=input_dimension))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

# optimizer = SGD(momentum=0.1, decay=1e-6, nesterov=True)
optimizer = Adadelta()
model.compile(optimizer, loss='mean_squared_error', metrics=['acc'])

batch_size = 32
model.fit_generator(generator(batch_size), epochs=10000, steps_per_epoch=batch_size)
score = model.evaluate_generator(generator(batch_size), steps=1000)

# save model and weights
model.save('model.h5')

def convert_cidade(city):
    return string_to_float_array(city, input_dimension)


cidades = ['163456789123Florianópolis', '163456789123Concórdia', '163456789123Chapecó', '163456789123João Pessoa', '163456789123Pindamonhangaba']
values = np.zeros((len(cidades), input_dimension))
for i, c in enumerate(cidades):
    values[i] = convert_cidade(c)
prediction = model.predict(values)

# print(cidade + "nota mtm: + " + prediction)
print(prediction)


