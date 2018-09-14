FROM python:3.6
ADD . /app
WORKDIR /app

RUN pip install -e .

RUN pip install -r requirements.txt

RUN mkdir ~/.keras
RUN touch ~/.keras/keras.json
RUN echo "{\"image_dim_ordering\": \"th\", \"epsilon\": 1e-07, \"floatx\": \"float32\", \"backend\": \"theano\" }" >> ~/.keras/keras.json

RUN chmod 777 -R /app