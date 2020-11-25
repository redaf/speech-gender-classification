# Speech gender classification

## Build docker container

```
docker build -t voice .
```

## Run container

Windows

```
docker run -v "${pwd}:/home" voice /path/to/file.wav
```

Unix
```
docker run -v $(pwd):/home voice /path/to/file.wav
```
