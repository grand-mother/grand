# Enable graphical application in docker

## For Linux

### Solution 1: with $HOME/.Xauthority


[tuto](https://medium.com/@SaravSun/running-gui-applications-inside-docker-containers-83d65c0db110)

For a GUI Application to run, we need to have a XServer which is available as part of every Linux Desktop Environment, But within a Container we don’t have any XServer — so we will

    share the Host’s XServer with the Container by creating a volume
    --volume="$HOME/.Xauthority:/root/.Xauthority:rw"
    share the Host’s DISPLAY environment variable to the Container
    --env="DISPLAY"
    run container with host network driver with
    --net=host

```
docker run --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --name test_e  docker_image_with_app_graph
```


### Solution 2: with /tmp/.X11

[tuto](https://leimao.github.io/blog/Docker-Container-GUI-Display/)

```
$ xhost +
$ docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix firefox:0.0.1
$ xhost -

```
## For MacOS

[ROOT proposition](https://hub.docker.com/r/rootproject/root)

feedback...

## For Windows

[ROOT proposition](https://hub.docker.com/r/rootproject/root)

feedback...

# Dockerhub command

[repository grandlib](https://hub.docker.com/u/jcolley)

How push on DocherHub

```
docker tag grand_dev:toto jcolley/grandlib_dev:1.0
docker push jcolley/grandlib_ci:1.0
```