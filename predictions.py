from keras.models import Sequential, load_model
import numpy as np

input_dimension = 32


def string_to_float_array(string, size):
    x = np.zeros(size)
    array = string.encode('utf-8')
    for i, b in enumerate(array):
        if i >= size:
            break
        x[i] = float(b)
    return x


def convert_cidade(city):
    return string_to_float_array(city, input_dimension)


model = load_model('model.h5')

cidades = ['163456789123Florianópolis', '163456789123Concórdia', '163456789123Chapecó', '163456789123João Pessoa', '163456789123Pindamonhangaba']
values = np.zeros((len(cidades), input_dimension))
for i, c in enumerate(cidades):
    values[i] = convert_cidade(c)
prediction = model.predict(values)

print("Predições:")

# print(cidade + "nota mtm: + " + prediction)
for i, c in enumerate(cidades):
    print(c + " -> " + str(prediction[i]))
