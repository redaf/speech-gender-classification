# Speech gender classification

## Build docker container

```
docker build -t voice .
```

## Run container

```
docker run -v "${pwd}:/home" voice /path/to/file.wav
```
