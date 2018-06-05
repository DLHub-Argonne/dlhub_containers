FROM python:3.6
ADD github-key /root/.ssh/id_rsa
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip3 install parsl

RUN chmod 600 /root/.ssh/id_rsa && echo "IdentityFile /root/.ssh/id_rsa" >> /etc/ssh/ssh_config && echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config

RUN git clone git@github.com:ryanchard/dlhub-service.git

RUN rm /root/.ssh/id_rsa
#RUN ln -s dlhub-service/web/dlhub/ .
RUN mv dlhub-service/web/* .
