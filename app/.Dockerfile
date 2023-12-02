FROM python:3.10.4

WORKDIR /root/code

RUN pip3 install dash
RUN pip3 install dash_bootstrap_components
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install scikit-learn==1.3.2
RUN pip3 install dash-bootstrap-components[pandas]
RUN pip3 install cloudpickle
RUN pip3 install mlflow

COPY ./code /root/code/
CMD tail -f /dev/null