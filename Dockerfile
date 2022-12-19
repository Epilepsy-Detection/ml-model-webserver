#Using the base image with python 3.7
FROM python:3.7

#Set our working directory as app
WORKDIR /app  
#Installing python packages pandas, scikit-learn and gunicorn
RUN pip install pandas scikit-learn flask gunicorn tensorflow numpy

# Copy the models directory and server.py files
ADD ./models ./models
ADD server.py server.py

#Exposing the port 2000 from the container
EXPOSE 2000 

#Starting the python application
CMD ["gunicorn", "--bind", "0.0.0.0:2000", "server:app"]