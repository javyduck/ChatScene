docker run \
    -it \
    --privileged \
    --name safebench \
    --gpus all \
    --net host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    safebench/safebench \
    /bin/bash
