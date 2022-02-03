# Enable graphical application in docker

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