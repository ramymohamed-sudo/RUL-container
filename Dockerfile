FROM python:latest
RUN apt-get update -y
RUN apt-get install vim -y

WORKDIR /usr/src/app

# WORKDIR ./
# COPY . . 

COPY requirements.txt ./
COPY run.sh ./
RUN chmod a+x run.sh

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install websocket-client
 
COPY . .

# CMD ["./run.sh"]
CMD ["python","./load_model.py"]
# ENTRYPOINT ["/bin/bash"]
# CMD ["run.sh"]
