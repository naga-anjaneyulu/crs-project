# base image 
FROM node:12

#set working directory 
RUN mkdir crs-ui

#Copy files to docker
COPY . /crs-ui
WORKDIR /crs-ui
RUN  npm install
EXPOSE 3000

CMD [ "npm","start"]

